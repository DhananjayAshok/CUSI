"""Shared factories and flags (curiosity_plan §3.2.4). run_explore.py ppo and
the pre-exploration search all take the same flag names, so their novelty numbers compare.

    @click.command()
    @state_options(scorer=True)        # adds the flags below
    def cmd(..., **kwargs):
        config = state_config(kwargs)
        encoder = build_encoder(config=config, env_name=env_name, parameters=parameters)
        archive = build_archive(config=config, encoder=encoder)
        scorer = build_scorer_from_config(config=config, env_name=env_name, encoder=encoder)

Flags: --image_embedder {random_patch,cnn,siglip} (required), --encoder_model (siglip),
--embedder_load_path (trained cnn / fine-tuned siglip), --text_embedder {dense,tfidf,overlap,none}
(required), --text_embedder_model (dense), --w_image, --similarity_metric, and with scorer=True:
--novelty_scorer (required), --region_alpha, --world_model_load_path.
Models are never defaulted.
"""
from typing import Any, Optional
import click
from cusi.state.archive import NoveltyArchive
from cusi.state.encoders import IMAGE_EMBEDDERS, TEXT_EMBEDDERS, build_image_embedder, build_text_embedder
from cusi.state.scorers import NOVELTY_SCORERS, build_scorer
from cusi.state.state import SIMILARITY_METRICS, StateEncoder

STATE_KEYS = ("image_embedder", "encoder_model", "embedder_load_path", "text_embedder", "text_embedder_model",
              "w_image", "similarity_metric")
SCORER_KEYS = ("novelty_scorer", "region_alpha", "world_model_load_path")


def state_options(*, scorer: bool = False):
    options = [
        click.option("--image_embedder", required=True, type=click.Choice(IMAGE_EMBEDDERS)),
        click.option("--encoder_model", default=None, help="SigLIP 2 model id (required with --image_embedder siglip)."),
        click.option("--embedder_load_path", default=None, help="Trained cnn / fine-tuned siglip checkpoint dir."),
        click.option("--text_embedder", required=True, type=click.Choice(TEXT_EMBEDDERS)),
        click.option("--text_embedder_model", default=None, help="Sentence model id (required with --text_embedder dense)."),
        click.option("--w_image", default=0.5, type=float, help="Image vs text weight in the similarity (1 if text is none)."),
        click.option("--similarity_metric", default="cosine", type=click.Choice(SIMILARITY_METRICS)),
    ]
    if scorer:
        options += [
            click.option("--novelty_scorer", required=True, type=click.Choice(NOVELTY_SCORERS)),
            click.option("--region_alpha", default=0.5, type=float, help="Region weight in the combination scorer."),
            click.option("--world_model_load_path", default=None, help="Trained world model dir (world_model scorer)."),
        ]

    def decorator(f):
        for option in reversed(options):
            f = option(f)
        return f
    return decorator


def state_config(kwargs: dict) -> dict:
    """Pop the shared flags out of a click command's kwargs."""
    return {k: kwargs.pop(k) for k in STATE_KEYS + SCORER_KEYS if k in kwargs}


def build_encoder(*, config: dict, env_name: str, device: Optional[str] = None,
                  parameters: dict[str, Any] = None) -> StateEncoder:
    if config["image_embedder"] == "siglip" and not config.get("encoder_model"):
        raise click.UsageError("--image_embedder siglip needs --encoder_model")
    if config["text_embedder"] == "dense" and not config.get("text_embedder_model"):
        raise click.UsageError("--text_embedder dense needs --text_embedder_model")
    image = build_image_embedder(image_embedder=config["image_embedder"], env_name=env_name,
                                 encoder_model=config.get("encoder_model"),
                                 embedder_load_path=config.get("embedder_load_path"), device=device,
                                 parameters=parameters)
    text = build_text_embedder(text_embedder=config["text_embedder"],
                               text_embedder_model=config.get("text_embedder_model"), device=device)
    return StateEncoder(image=image, text=text, w_image=config.get("w_image", 0.5),
                        metric=config.get("similarity_metric", "cosine"))


def build_archive(*, encoder: StateEncoder, cell_threshold: Optional[float] = None,
                  max_size: int = 10_000) -> NoveltyArchive:
    return NoveltyArchive(metric=encoder.metric, text_embedder=encoder.text, w_image=encoder.w_image,
                          cell_threshold=cell_threshold, max_size=max_size)


def build_scorer_from_config(*, config: dict, env_name: str, encoder: StateEncoder, device: Optional[str] = None):
    return build_scorer(novelty_scorer=config["novelty_scorer"], env_name=env_name,
                        region_alpha=config.get("region_alpha", 0.5),
                        world_model_load_path=config.get("world_model_load_path"), encoder=encoder, device=device)
