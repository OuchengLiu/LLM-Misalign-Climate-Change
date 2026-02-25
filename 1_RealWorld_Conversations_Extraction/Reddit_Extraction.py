import praw
import json
import yaml
from tqdm import tqdm
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration helpers                                                       #
# --------------------------------------------------------------------------- #
def load_cfg(section: str = "1_RealWorld_Conversations_Extraction") -> dict:
    """Return the requested section from Configs.yaml (repo root assumed)."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg_file = repo_root / "Configs.yaml"
    with cfg_file.open("r", encoding="utf-8") as fp:
        raw_cfg = yaml.safe_load(fp) or {}
    return raw_cfg.get(section, {})

CFG = load_cfg()
MCFG = CFG.get("Reddit_Extraction", {})

CLIENT_ID = MCFG.get("Client_ID", "")
CLIENT_SECRET = MCFG.get("Client_Secret", "")
USER_AGENT = MCFG.get("User_Agent", "")
OUTPUT_FILE = Path(MCFG.get("Output_File", "")).expanduser().resolve()
# If your config has a community list, use it; otherwise, manually specify:
COMMUNITY_LIST = MCFG.get("Community_List", ["climatechange", "climate", "climate_science", "GlobalClimateChange"])

# ====== Reddit API Configuration ======
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

all_posts = []

for community in COMMUNITY_LIST:
    subreddit = reddit.subreddit(community)
    collected_ids = set()
    community_posts = []

    # Helper to add unique posts with question marks only
    def try_add(submission):
        if submission.id in collected_ids:
            return
        if "?" in submission.title or "?" in submission.selftext:
            post = {
                "reddit_id": submission.id,  # Save Reddit's post id for deduplication
                "title": submission.title,
                "description": submission.selftext,
                "upvotes": submission.score,
                "num_comments": submission.num_comments,
                "created_utc": submission.created_utc,
            }
            community_posts.append(post)
            collected_ids.add(submission.id)

    # Fetch posts from 'new'
    for submission in tqdm(subreddit.new(limit=None), desc=f"{community} - new"):
        try_add(submission)
    # Fetch posts from 'hot'
    for submission in tqdm(subreddit.hot(limit=None), desc=f"{community} - hot"):
        try_add(submission)
    # Fetch posts from 'top'
    for submission in tqdm(subreddit.top(limit=None), desc=f"{community} - top"):
        try_add(submission)

    # Sort posts by creation time in ascending order
    community_posts = sorted(community_posts, key=lambda x: x["created_utc"])
    # Add community-specific id: communityname+numid
    for idx, post in enumerate(community_posts, 1):
        post["id"] = f"{community}_{idx}"

    all_posts.extend(community_posts)

# Save results as JSONL: one JSON per line
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for post in all_posts:
        out = {
            "id": post["id"],
            "title": post["title"],
            "description": post["description"],
            "upvotes": post["upvotes"],
            "num_comments": post["num_comments"],
        }
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

print(f"Saved {len(all_posts)} question posts from all communities to {OUTPUT_FILE}")
