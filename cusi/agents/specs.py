"""
The environments as the pipeline uses them: scenes, the final agent and prompt domains.
"""
import threading
from dataclasses import dataclass, field
from typing import Any, Callable
from cusi.practice.prompts import ANDROID_DOMAIN, WEB_DOMAIN, Domain, gameboy_domain
from cusi.agents.agent import make_agent

ENV_NAMES = ("gameboy", "android", "web")
GAMEBOY_GAME = "pokemon_red"


#: The start states of GameBoyWorlds' pokemon_red benchmark tasks (gameboy_worlds.utils.get_benchmark_tasks), in
#: benchmark order: exploration starts where the test tasks do.
GAMEBOY_TEST_STATES = (
    "location_viridian_city_starting_charmander", "location_mt_moon_complete", "location_bills_house_complete",
    "location_viridian_forest_starting_charmander", "location_cinnabar_island_complete",
    "trainer_brock_starting_charmander", "trainer_lass_starting_charmander", "encounter_pidgey_starting_charmander",
    "encounter_pikachu_starting_charmander", "location_pewter_city_starting_charmander",
    "location_pewter_pokemart_starting_charmander", "complete_encounter_seaking", "in_bike_store",
    "in_badge_info_man_house", "in_cerulean_map_house", "battle_blue_cerulean_bridge", "location_cinnabar_mansion",
    "location_cinnabar_facing_water", "location_cinnabar_mart", "location_ghost_tower", "outside_name_rater",
    "letter_house", "outside_elite_four", "close_to_pond", "border_stop", "pokemon_fan_house", "in_silf_co",
    "outside_robbed_house", "outside_daycare", "near_saffron_sign", "inside_seafoam_islands", "test_cinnabar_monk",
    "inside_saffron_pokemon_center", "next_to_sign", "battle_pidgey", "in_fuschia_gym", "in_viridian_gym",
    "bridge_water", "in_game_house", "pidgeotto_fight")
#: AndroidWorld's apps (android_world.env.adb_utils._PATTERN_TO_ACTIVITY), by the short names the scenes use.
ANDROID_PACKAGES = {"markor": "net.gsantner.markor", "settings": "com.android.settings",
                    "broccoli": "com.flauschcode.broccoli", "expense": "com.arduia.expense",
                    "calendar": "com.simplemobiletools.calendar.pro", "sms": "com.simplemobiletools.smsmessenger",
                    "clock": "com.google.android.deskclock", "gallery": "com.simplemobiletools.gallery.pro",
                    "retro": "code.name.monkey.retromusic", "chrome": "com.android.chrome",
                    "camera": "com.android.camera2", "contacts": "com.google.android.contacts",
                    "vlc": "org.videolan.vlc", "osmand": "net.osmand", "audio_recorder": "com.dimowner.audiorecorder",
                    "files": "com.google.android.documentsui", "draw": "com.simplemobiletools.draw.pro",
                    "joplin": "net.cozic.joplin", "tasks": "org.tasks", "opentracks": "de.dennisguse.opentracks",
                    "clipper": "ca.zgrs.clipper"}
#: Screens Settings opens from other packages (its search, Google settings, Digital Wellbeing, text-to-speech, SIM,
#: storage, sounds, wallpaper, legal notices): seen in the Settings scenes of the unscoped Android searches.
SETTINGS_EXTRAS = ("com.google.android.settings.intelligence", "com.google.android.gms",
                   "com.google.android.apps.wellbeing", "com.google.android.tts", "com.android.phone",
                   "com.android.storagemanager", "com.google.android.soundpicker", "com.google.android.apps.wallpaper",
                   "com.android.htmlviewer")


@dataclass(frozen=True)
class Scene:
    name: str
    kwargs: dict = field(default_factory=dict)


def scene_for(*, scene: Scene, scoped: bool) -> Scene:
    """The scene as a run uses it: unscoped (runs from before scenes were kept on their website or app) without its
    scope."""
    return scene if scoped else Scene(scene.name, {k: v for k, v in scene.kwargs.items() if k != "scope"})


@dataclass(frozen=True)
class EnvSpec:
    name: str
    domain: Domain
    scenes: dict
    max_steps: int               # step budget per leg (attempt and practice)
    max_workers: int             # concurrent environment instances
    make_env: Callable
    make_agent: Callable         # make_agent(env=, vlm=, parameters=) -> the env's final agent
    env_description: str         # the env's own env_description, for code without a live env (cusi.search.annotate)


def _gameboy_env(*, scene: Scene, worker: int, parameters: dict):
    # Imports are lazy: GameBoyRL's and WebVoyager's top-level `utils` modules cannot share a process.
    from cusi.envs.gameboy import GameBoyPlayEnv
    return GameBoyPlayEnv(game=GAMEBOY_GAME, mode="free_play", parameters=parameters, **scene.kwargs)


def _android_env(*, scene: Scene, worker: int, parameters: dict):
    from cusi.envs.android_world import AndroidEmulator, AndroidPlayEnv
    port, grpc = 5554 + 2 * worker, 8554 + worker   # must match scripts/android_emulator.sh
    emulator = AndroidEmulator(port=port, grpc_port=grpc, snapshots=True, parameters=parameters)
    # The agent re-observes before each decision (as M3A does), so the env does not wait for the screen to settle.
    return AndroidPlayEnv(mode="free_play", console_port=port, grpc_port=grpc, emulator=emulator,
                          reset_mode="snapshot", stabilize_after_action=False, parameters=parameters, **scene.kwargs)


def _web_env(*, scene: Scene, worker: int, parameters: dict):
    from cusi.envs.webvoyager import WebVoyagerPlayEnv
    return WebVoyagerPlayEnv(mode="free_play", fast_waits=True, parameters=parameters, **scene.kwargs)


def _agent_factory(env_name: str) -> Callable:
    def make(*, env, vlm, parameters: dict):
        return make_agent(env_name=env_name, env=env, vlm=vlm, parameters=parameters)
    return make


ENV_SPECS = {
    "gameboy": EnvSpec(
        name="gameboy", domain=gameboy_domain(game=GAMEBOY_GAME),
        # One scene per start state of the test set (GameBoyWorlds' pokemon_red benchmark: 49 tasks, 40 states).
        scenes={state: Scene(state, {"init_state": state}) for state in GAMEBOY_TEST_STATES},
        max_steps=50, max_workers=8, make_env=_gameboy_env, make_agent=_agent_factory("gameboy"),
        env_description=f"the GameBoy game {GAMEBOY_GAME}"),
    # AndroidWorld budgets 10 steps per unit of task complexity.
    "android": EnvSpec(
        name="android", domain=ANDROID_DOMAIN,
        # One scene per app of the test set (AndroidWorld's templates), started in the app (start_app: tasks start on
        # the home screen, which would make the scenes look identical), its data seeded by one of the app's
        # templates. Each is kept in its app (cusi.envs.android_world): its package, the system screens apps share,
        # and the apps it is paired with in the test set's multi-app templates. Dialer (only in OpenAppTaskEval) and
        # Clipper (only paired) have no scene of their own.
        scenes={name: Scene(name, {"task": task, "start_app": app, "app_label": label,
                                   "scope": {"packages": packages}})
                for name, task, app, label, packages in (
            ("markor", "MarkorCreateNote", "markor", "Markor",
             [ANDROID_PACKAGES[a] for a in ("markor", "expense", "broccoli", "sms", "clipper", "gallery", "vlc")]),
            ("settings", "SystemWifiTurnOff", "Settings", "Settings", [ANDROID_PACKAGES["settings"], *SETTINGS_EXTRAS]),
            ("broccoli", "RecipeAddSingleRecipe", "broccoli app", "Broccoli",
             [ANDROID_PACKAGES[a] for a in ("broccoli", "gallery", "markor")]),
            ("expense", "ExpenseAddSingle", "pro expense", "Pro Expense",
             [ANDROID_PACKAGES[a] for a in ("expense", "gallery", "markor")]),
            ("calendar", "SimpleCalendarAddOneEvent", "simple calendar pro", "Simple Calendar Pro",
             [ANDROID_PACKAGES["calendar"]]),
            ("sms", "SimpleSmsSend", "simple sms messenger", "Simple SMS Messenger",
             [ANDROID_PACKAGES[a] for a in ("sms", "markor", "clipper")]),
            ("clock", "ClockTimerEntry", "clock", "Clock", [ANDROID_PACKAGES["clock"]]),
            ("gallery", "SaveCopyOfReceiptTaskEval", "simple gallery pro", "Simple Gallery Pro",
             [ANDROID_PACKAGES[a] for a in ("gallery", "expense", "markor", "broccoli")]),
            ("retro", "RetroCreatePlaylist", "retro music", "Retro Music", [ANDROID_PACKAGES["retro"]]),
            ("chrome", "BrowserMaze", "chrome", "Chrome", [ANDROID_PACKAGES["chrome"]]),
            ("camera", "CameraTakePhoto", "camera", "Camera", [ANDROID_PACKAGES["camera"]]),
            ("contacts", "ContactsAddContact", "Contacts", "Contacts", [ANDROID_PACKAGES["contacts"]]),
            ("vlc", "VlcCreatePlaylist", "vlc", "VLC", [ANDROID_PACKAGES[a] for a in ("vlc", "markor")]),
            ("osmand", "OsmAndFavorite", "osmand", "OsmAnd", [ANDROID_PACKAGES["osmand"]]),
            ("audio_recorder", "AudioRecorderRecordAudio", "audio recorder", "Audio Recorder",
             [ANDROID_PACKAGES["audio_recorder"]]),
            ("files", "FilesDeleteFile", "files", "Files", [ANDROID_PACKAGES["files"]]),
            ("draw", "SimpleDrawProCreateDrawing", "simple draw pro", "Simple Draw Pro", [ANDROID_PACKAGES["draw"]]),
            ("joplin", "NotesIsTodo", "joplin", "Joplin", [ANDROID_PACKAGES["joplin"]]),
            ("tasks", "TasksDueOnDate", "tasks", "Tasks", [ANDROID_PACKAGES["tasks"]]),
            ("opentracks", "SportsTrackerActivitiesOnDate", "open tracks", "OpenTracks",
             [ANDROID_PACKAGES["opentracks"]]),
        )},
        max_steps=20, max_workers=1, make_env=_android_env, make_agent=_agent_factory("android"),
        env_description="an Android phone"),
    "web": EnvSpec(
        name="web", domain=WEB_DOMAIN,
        # One scene per website of the test set (WebVoyager minus web_excluded_sites), each kept on its website
        # (cusi.envs.webvoyager.in_scope): its domain and subdomains, or a path under www.google.com.
        scenes={name: Scene(name, {"url": url, "site": site, "scope": scope}) for name, url, site, scope in (
            ("arxiv", "https://arxiv.org/", "arXiv", {"hosts": ["arxiv.org"]}),
            ("github", "https://github.com/", "GitHub", {"hosts": ["github.com"]}),
            ("apple", "https://www.apple.com/", "Apple", {"hosts": ["apple.com"]}),
            ("booking", "https://www.booking.com/", "Booking.com", {"hosts": ["booking.com"]}),
            ("coursera", "https://www.coursera.org/", "Coursera", {"hosts": ["coursera.org"]}),
            ("espn", "https://www.espn.com/", "ESPN", {"hosts": ["espn.com"]}),
            ("huggingface", "https://huggingface.co/", "Hugging Face", {"hosts": ["huggingface.co"]}),
            ("wolframalpha", "https://www.wolframalpha.com/", "Wolfram|Alpha", {"hosts": ["wolframalpha.com"]}),
            ("google_flights", "https://www.google.com/travel/flights/", "Google Flights",
             {"hosts": ["www.google.com"], "path": "/travel/flights"}),
            ("google_maps", "https://www.google.com/maps/", "Google Maps", {"hosts": ["www.google.com"], "path": "/maps"}),
        )},
        max_steps=15, max_workers=2, make_env=_web_env, make_agent=_agent_factory("web"),
        env_description="a web browser"),
}


class EnvPool:
    """Live environments reused across legs, one per worker; each worker thread uses its own index."""

    def __init__(self, *, spec: EnvSpec, parameters: dict[str, Any], scoped: bool = True) -> None:
        """scoped: keep each scene on its website or app (False for trees from unscoped searches)."""
        self.spec = spec
        self.scoped = scoped
        self._parameters = parameters
        self._envs: dict = {}
        self._lock = threading.Lock()

    def get(self, *, worker: int, scene: str):
        key = (worker, scene)
        with self._lock:
            env = self._envs.get(key)
            # One env per worker at a time: an Android emulator holds one scene snapshot.
            stale = [(k, e) for k, e in self._envs.items() if k[0] == worker and k != key]
            for k, _ in stale:
                del self._envs[k]
        for _, e in stale:
            e.close()
        if env is None:
            env = self.spec.make_env(scene=scene_for(scene=self.spec.scenes[scene], scoped=self.scoped),
                                     worker=worker, parameters=self._parameters)
            with self._lock:
                self._envs[key] = env
        return env

    def close(self) -> None:
        for env in self._envs.values():
            try:
                env.close()
            except Exception:
                pass
        self._envs = {}
