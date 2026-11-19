import uuid

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

COLLECTION = "documents"
DIM = 384
DENSE = "dense"
SPARSE = "bm25"
NAMESPACE = uuid.UUID("6f1b6b0e-6f7e-4c1e-9d55-0d5a3b6d6c11")

_model = _sparse = _client = _reranker = None


def get_model():
    global _model
    if _model is None:
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _model


def get_sparse():
    global _sparse
    if _sparse is None:
        _sparse = SparseTextEmbedding("Qdrant/bm25")
    return _sparse


def get_reranker():
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
    return _reranker


def get_client():
    global _client
    if _client is None:
        _client = QdrantClient(path="qdrant_data")
        if not _client.collection_exists(COLLECTION):
            _client.create_collection(
                COLLECTION,
                vectors_config={DENSE: models.VectorParams(size=DIM, distance=models.Distance.COSINE)},
                sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
    return _client


def _group_filter(groups):
    """Only chunks labelled with one of the user's groups. No groups means 'everyone' only."""
    groups = list(groups) if groups else ["everyone"]
    return models.Filter(
        must=[models.FieldCondition(key="group", match=models.MatchAny(any=groups))]
    )


def count_chunks(groups):
    return get_client().count(COLLECTION, count_filter=_group_filter(groups), exact=True).count


def delete_source(source):
    get_client().delete(
        COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
            )
        ),
    )


def add_chunks(source, chunks, group="everyone", context=""):
    """chunks is a list of (locator, text). Re-adding a file replaces its old chunks.
    context is a one-line description of the document, searched together with each chunk."""
    delete_source(source)
    client, model, sparse = get_client(), get_model(), get_sparse()
    for start in range(0, len(chunks), 64):
        part = chunks[start:start + 64]
        texts = [(context + "\n" + text) if context else text for _, text in part]
        dense = list(model.embed(texts, batch_size=8))
        sp = list(sparse.embed(texts, batch_size=8))
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(NAMESPACE, f"{source}:{start + i}")),
                vector={
                    DENSE: d.tolist(),
                    SPARSE: models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
                },
                payload={"source": source, "locator": loc, "text": text, "group": group, "context": context},
            )
            for i, ((loc, text), d, s) in enumerate(zip(part, dense, sp))
        ]
        client.upsert(COLLECTION, points=points)

def _filter(groups, sources=None):
    """Group filter, optionally narrowed to specific files. It can only narrow, never widen."""
    groups = list(groups) if groups else ["everyone"]
    must = [models.FieldCondition(key="group", match=models.MatchAny(any=groups))]
    if sources:
        must.append(models.FieldCondition(key="source", match=models.MatchAny(any=list(sources))))
    return models.Filter(must=must)

def search(query, k=5, candidates=30, rerank=True, groups=None, sources=None):
    flt = _filter(groups, sources)
    q = next(iter(get_model().query_embed(query))).tolist()
    s = next(iter(get_sparse().query_embed(query)))
    points = get_client().query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=q, using=DENSE, limit=30, filter=flt),
            models.Prefetch(
                query=models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
                using=SPARSE,
                limit=30,
                filter=flt,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=flt,
        limit=max(candidates, k),
        with_payload=True,
    ).points
    if rerank and len(points) > 1:
        scores = list(get_reranker().rerank(
            query, [(p.payload.get("context", "") + "\n" + p.payload["text"]) for p in points]
        ))
        ranked = sorted(zip(scores, points), key=lambda pair: pair[0], reverse=True)
        points = [p for _, p in ranked]
    return points[:k]