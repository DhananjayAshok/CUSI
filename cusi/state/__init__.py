"""cusi.state: what curiosity PPO and the pre-exploration search share (curiosity_plan §3.2).
Inference only. Three layers:

    encoders/   image embedders (random_patch, cnn, siglip) + text embedders (none, overlap, tfidf,
                dense) -> StateEncoder -> StateEmbedding, with a weighted similarity
    archive.py  NoveltyArchive: add / dedup / compaction / copy-restore / cells / region stores
    scorers/    embedding, region, combination, world_model: score(prev, action, next, archive)

Training (cnn, siglip fine-tuning, world models) lives in cusi.explore; lifetime and reward
shaping live with each consumer. Interface changes need a note in plans/curiosity_plan.md and
plans/skill_discovery.md.
"""
from cusi.state.archive import NoveltyArchive
from cusi.state.canvas import CANVAS, frame_for_embedding, to_canvas
from cusi.state.encoders import build_image_embedder, build_text_embedder
from cusi.state.factories import build_archive, build_encoder, build_scorer_from_config, state_config, state_options
from cusi.state.scorers import build_scorer
from cusi.state.state import StateEmbedding, StateEncoder, StateRecord
from cusi.state.text import element_lines
