"""
Transition descriptions: after an expansion, a VLM sees its whole sub-trajectory (the start screen, then the screen
after every action) and describes what happened.
"""
from typing import Optional
from cusi.search.context import action_line

DESCRIBE_PROMPT = """These {n_images} screenshots show {env_description}, in order: the starting screen, then the \
screen after each of the {n_actions} actions below.

Actions, in order:
{actions}

Describe in two or three sentences what happens over this sequence: where it starts, what the actions did \
(including any that had no visible effect), and where it ends. Name the screens, menus, pages, places or features \
reached. Reply with the description only."""


class TransitionDescriber:
    def __init__(self, *, vlm, max_new_tokens: int = 200, temperature: float = 0.0, max_chars: int = 600) -> None:
        self.vlm, self.max_new_tokens, self.temperature, self.max_chars = vlm, max_new_tokens, temperature, max_chars
        self.n_calls = 0

    def prompt(self, *, env_description: str, steps: list) -> str:
        actions = "\n".join(f"{i + 1}. {action_line(action=s.action, valid=s.info.get('valid', True))}"
                            for i, s in enumerate(steps))
        return DESCRIBE_PROMPT.format(n_images=len(steps) + 1, env_description=env_description,
                                      n_actions=len(steps), actions=actions)

    def __call__(self, *, env_description: str, start_frame, steps: list) -> Optional[str]:
        """start_frame and each step's frame: EncodedImage or FrameRef (anything with .pil())."""
        if not steps:
            return None
        images = [start_frame.pil()] + [s.frame.pil() for s in steps]
        out = self.vlm.infer(texts=self.prompt(env_description=env_description, steps=steps), images=images,
                             max_new_tokens=self.max_new_tokens, temperature=self.temperature)
        self.n_calls += 1
        text = " ".join(str(out["output"]).split())
        return text[:self.max_chars] or None
