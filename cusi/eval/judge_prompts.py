"""
Per-environment prompts for the two task judges (plans/task_judge_prompts.md), one shared scale on how alike two tasks
are in the environment mechanics and functions they use and the capabilities they need, not in their wording:
- coverage (cusi.eval.task_coverage): a tree's proposed task (B) against a benchmark task (A); examples from the test
  set (A: benchmark tasks; B: proposals from our trees, or written where none sat at that level);
- internal (cusi.eval.task_diversity, dedup): two proposed tasks; examples from made-up environments only, with no
  test-set content.
The judges see only the two task texts and the env description (no site, app or start-state context).

Judge outputs go to judge_<model>_<PROMPT_VERSION>/, so judgments under another prompt never mix.
"""

PROMPT_VERSION = "v2"

#: A benchmark task is covered by a proposal scored at least this.
COVERED_AT = 7
#: Two proposals exercise the same functionality when scored at least this (diversity's functions@7).
SAME_FUNCTION_AT = 7
#: Two proposals are the same task (merge them) when scored at least this.
DEDUP_AT = 9

ENV_DESCRIPTIONS = {
    "web": "a web browser",
    "android": "an Android phone",
    "gameboy": "a GameBoy game (Pokémon Red or The Legend of Zelda: Link's Awakening)",
}

SCALE = """Score how similar the two tasks are in what doing them takes: the mechanics and functions of the environment \
used to complete them and the capabilities they require. Judge the work of executing each task, not its wording.
- 10-9: exact or near-exact paraphrases of each other.
- 8-7: fundamentally the same task: the same objective, the same mechanics used, the same capabilities required. Only \
entity names or exact details differ (another city, date, name, amount, item or character), and an agent able to do \
one could do the other the same way: they are interchangeable for task execution.
- 6-5: similar but different: they share part of the work, but a key mechanic differs, or the objective differs (one \
stops early, goes further, ends in a different state, or adds or drops a requirement).
- 4-3: the same app, website or area of the game, but mostly different mechanics and a different objective; at most a \
small shared part (the same entry screen or menu).
- 2-1: different mechanics and different objectives, or a different app, website or game.
Use only what the two texts state: do not assume a connection they do not make."""

REPLY_FORMAT = """Reply in exactly this format:
Reason: <one sentence>
Score: <an integer from 1 to 10>"""

# ------------------------------------------------------------------ coverage examples (from the test set)
# A: benchmark tasks (split ids in plans/task_judge_prompts.md); B: real proposals from our trees unless noted there.

COVERAGE_EXAMPLES = {
    "web": """- Task A: "Identify bus stops in Ypsilanti, MI, list three of them."
  Task B: "List three bus stops located in Ypsilanti, Michigan."
  Reason: The same lookup on a map site for the same place and the same answer, reworded.
  Score: 10
- Task A: "Search for the cheapest round-trip flights from Bangkok to Madrid, leaving on February 26, 2024, and \
returning on February 28, 2024, and provide options under $1000."
  Task B: "Find the cheapest round-trip flights from New York to London departing on October 10th and returning on \
October 20th for one adult in economy."
  Reason: The same flight search (origin, destination, two dates, sort by price) with other cities, dates and price \
details; an agent doing one would do the other the same way.
  Score: 8
- Task A: "Find a Beginner's Spanish Specialization on Coursera and show all the courses in this Specialization."
  Task B: "Use the filters to find a "Beginner" level course that is a "Specialization" and provide the title of the \
course."
  Reason: Both search the course catalogue with the level and type filters, but B sets no subject and stops at a \
title, while A opens the Specialization and lists its courses.
  Score: 6
- Task A: "Find the Resolve merge conflicts course in GitHub Skills and what actions learners will perform in this \
course."
  Task B: "Navigate to the Courses section of GitHub Learn and return to the Verified Proficiency page."
  Reason: The same site's learning area, but B only moves between sections, while A finds a specific course and reads \
what it involves.
  Score: 4
- Task A: "Give the geomagnetic field on June 20, 2023 in Oslo."
  Task B: "Access the local weather forecast to determine the outlook for the next 7 days."
  Reason: A computes a scientific quantity for a past date and place; B reads a weather forecast: different functions \
and objectives.
  Score: 2""",
    "android": """- Task A: "Run the stopwatch."
  Task B: "In the Clock app, start the stopwatch."
  Reason: The same action, reworded.
  Score: 10
- Task A: "Create a new contact for Hugo Pereira. Their number is +13920741751."
  Task B: "In the Contacts app, create a contact with the first name "Jane", last name "Doe", and the phone number \
"555-0123"."
  Reason: The same contact-creation flow with the same fields (name and number); only the values differ.
  Score: 8
- Task A: "Run the stopwatch."
  Task B: "In the Clock app, start the stopwatch and then stop it after a few seconds."
  Reason: The same feature, but B ends with the stopwatch stopped while A needs it running: the end state differs.
  Score: 6
- Task A: "In Simple Calendar Pro, create a calendar event on 2023-10-29 at 13h with the title 'Call with the Team' \
and the description 'We will understand upcoming project milestones.'. The event should last for 60 mins."
  Task B: "Using the Calendar app, find the date of the "Meeting" that occurs in the second half of October and state \
the day of the month."
  Reason: Both use the calendar, but A creates an event with its fields while B looks one up and reads its date.
  Score: 4
- Task A: "Add a favorite location marker for 47.1303814, 9.5930117 in the OsmAnd maps app."
  Task B: "In the OpenTracks app, start a new recording session by using the Record button."
  Reason: Different apps and different functions: saving a map location versus starting an activity recording.
  Score: 2""",
    "gameboy": """- Task A: "Defeat Gym Leader Brock"
  Task B: "Defeat Brock's Pokémon to win the gym badge."
  Reason: The same battle against the same leader with the same outcome, reworded.
  Score: 10
- Task A: "go inside the shop"
  Task B: "enter the trendy game house"
  Reason: The same mechanic, walking to a building in the village and entering it; only the building differs.
  Score: 8
- Task A: "Try to purchase a bike"
  Task B: "Talk to the person behind the counter."
  Reason: Talking to the clerk is part of A, but B names neither the bike shop nor a purchase, so its objective falls \
short of A's.
  Score: 6
- Task A: "Sell exactly 1 Psychic TM"
  Task B: "Throw away a Poké Ball"
  Reason: Both handle an item from the bag, but A sells it to a shop for money while B discards it from the menu: \
different mechanics and objectives.
  Score: 4
- Task A: "go and see the pond"
  Task B: "talk to the shopkeeper inside the shop"
  Reason: Walking to a landmark outdoors versus entering a building and talking to someone: different mechanics and \
objectives.
  Score: 2""",
}

# ------------------------------------------------------------------ internal examples (made-up environments only)

INTERNAL_EXAMPLES = {
    "web": """- Task A: "On the recipe site, find the vegan lasagna recipe and report its total cooking time."
  Task B: "Look up how long the vegan lasagna recipe on the recipe site takes to cook in total."
  Reason: The same lookup and the same answer, reworded.
  Score: 10
- Task A: "On the bike-rental site, find an e-bike available in Porto on 3 May and report its daily price."
  Task B: "On the bike-rental site, find a cargo bike available in Lyon on 12 June and report its daily price."
  Reason: The same availability search and price lookup with another bike type, city and date.
  Score: 8
- Task A: "On the bike-rental site, find an e-bike available in Porto on 3 May and report its daily price."
  Task B: "On the bike-rental site, search for e-bikes in Porto and filter the results to bikes with a child seat."
  Reason: The same search, but B applies a filter and stops, while A picks a date and reads a price.
  Score: 6
- Task A: "Find the opening hours of the bike-rental site's Porto branch."
  Task B: "Create an account on the bike-rental site with a new email address."
  Reason: The same site, but reading branch information and signing up use different features.
  Score: 4
- Task A: "On the bike-rental site, report the daily price of an e-bike in Porto."
  Task B: "On the recipe site, save the vegan lasagna recipe to your favourites."
  Reason: Different sites and different functions.
  Score: 2""",
    "android": """- Task A: "In the budget app, add an expense of $12.50 for "Lunch" in the Food category."
  Task B: "Record a $12.50 "Lunch" expense under Food in the budget app."
  Reason: The same entry with the same values, reworded.
  Score: 10
- Task A: "In the budget app, add an expense of $12.50 for "Lunch" in the Food category."
  Task B: "In the budget app, add an expense of $48.00 for "Train ticket" in the Transport category."
  Reason: The same add-expense form and fields; only the values differ.
  Score: 8
- Task A: "In the budget app, add an expense of $12.50 for "Lunch" in the Food category."
  Task B: "In the budget app, change the amount of the existing "Lunch" expense to $15.00."
  Reason: The same expense form, but B edits an existing entry instead of creating one.
  Score: 6
- Task A: "In the budget app, add an expense of $12.50 for "Lunch" in the Food category."
  Task B: "In the budget app's settings, change the displayed currency to euros."
  Reason: The same app, but adding an entry and changing a setting use different features.
  Score: 4
- Task A: "In the budget app, add an expense of $12.50 for "Lunch" in the Food category."
  Task B: "In the notes app, pin the note titled "Groceries"."
  Reason: Different apps and different functions.
  Score: 2""",
    "gameboy": """- Task A: "Talk to the blacksmith in his forge"
  Task B: "Speak with the blacksmith inside the forge"
  Reason: The same action with the same character, reworded.
  Score: 10
- Task A: "Talk to the blacksmith in his forge"
  Task B: "Talk to the baker inside the bakery"
  Reason: The same mechanic, entering a building and talking to the person there; only the character and building \
differ.
  Score: 8
- Task A: "Buy a wooden shield from the blacksmith"
  Task B: "Talk to the blacksmith"
  Reason: Talking to him is the first part of A, but A also needs the shop menu, money and an item bought.
  Score: 6
- Task A: "Buy a wooden shield from the blacksmith"
  Task B: "Equip the wooden shield from the inventory"
  Reason: The same item, but buying at a shop and equipping from the inventory menu are different mechanics.
  Score: 4
- Task A: "Buy a wooden shield from the blacksmith"
  Task B: "Defeat the slime in the forest clearing"
  Reason: Shopping in the village versus fighting outdoors: different mechanics and objectives.
  Score: 2""",
}


def _escape(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


#: Coverage prompts by env, to .format(test=..., proposed=...).
COVERAGE_PROMPTS = {
    env: (f"Task A is a task from a benchmark for an agent operating {_escape(ENV_DESCRIPTIONS[env])}. Task B is a task "
          f"proposed while exploring the same environment. {_escape(SCALE)}\n\nExamples:\n"
          f"{_escape(COVERAGE_EXAMPLES[env])}\n\nTask A: {{test}}\nTask B: {{proposed}}\n\n{_escape(REPLY_FORMAT)}")
    for env in ENV_DESCRIPTIONS}

#: Internal prompts by env, to .format(a=..., b=...).
INTERNAL_PROMPTS = {
    env: (f"Task A and Task B are two tasks for an agent operating {_escape(ENV_DESCRIPTIONS[env])}. {_escape(SCALE)}"
          f"\n\nExamples (from other, made-up environments):\n{_escape(INTERNAL_EXAMPLES[env])}\n\n"
          f"Task A: {{a}}\nTask B: {{b}}\n\n{_escape(REPLY_FORMAT)}")
    for env in ENV_DESCRIPTIONS}
