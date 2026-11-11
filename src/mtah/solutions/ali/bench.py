"""A benchmark on OTHER games: Metrica Sports' public sample data, in our exact bundle format.

Why it exists. The rules forbid training, tuning or choosing between two versions
on the public 5 minutes, and forbid hand labels on BAR-ATH for any purpose. Labels on OTHER games
are allowed in any amount, provided the organisers can access them and the technical document
explains them. Metrica's three public sample games (https://github.com/metrica-sports/sample-data,
"please acknowledge the source") come with human event data, so this module:

1. downloads them (pinned commit, sha256 checked); nothing about BAR-ATH is involved;
2. derives frame-level H/A/D ground truth from the events, mechanically, the way the organisers
   derived theirs from corrected tracking (see `derive_labels`);
3. degrades the clean player tracking towards the automatic ATD of the challenge: 46 track slots
   with fragments, gaps and identity switches, no player identities, a little noise (the ball
   already goes missing on ~40 % of frames in these games, as in BAR-ATH);
4. writes a bundle in the official file formats, so `MatchData`, every solution and the official
   scorer run on it unchanged:

    python -m mtah.solutions.ali.bench                     # <cache>/bench/metrica_{1,2,3}
    mtah run ali -d .cache/bench/metrica_1                  # scored against its ground truth

The bundle reuses the synthetic layout (`synthetic/meta.json`, `synthetic/gt_full.csv`) so the
rest of the code needs no change; `meta.json` says where it really comes from. No Smart Tagging and no
homography exist for these games: a solution must not require them to be scored here.

Second source, closer still to how the challenge's ground truth is made: IDSSE (Bassek et al.,
Scientific Data 2025, CC-BY 4.0, https://doi.org/10.6084/m9.figshare.28196177), seven Bundesliga
matches whose DFL position data carry, on every frame, `BallStatus` (alive / dead) and
`BallPossession` (home / away) from quality-controlled tracking: the same mechanical derivation
the organisers describe ("D where the corrected tracking marks the ball dead, otherwise the team the
tracking names as owning it"). No label has to be derived from events there.

    python -m mtah.solutions.ali.bench --game dfl_J03WMX    # or --dataset dfl / metrica / all
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import typer

from mtah.labels import A, Codes, D, H, X
from mtah.paths import cache_dir
from mtah.spec import MatchSpec

BENCH_VERSION = 1
FPS = 25
PITCH = (105.0, 68.0)  # README of the sample data
N_SLOTS = 46
UPSTREAM = "https://raw.githubusercontent.com/metrica-sports/sample-data"
COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
FILES: dict[int, dict[str, str]] = {
    1: {
        "Sample_Game_1_RawEventsData.csv": "097dda6345e17390dca095331458fd00e47960c480c76f4356fd4edf2575ef05",
        "Sample_Game_1_RawTrackingData_Away_Team.csv": "0a3df68a92af0290ede472e62de3b6489ddee29223da09bdf0cf76c8e0ab0618",
        "Sample_Game_1_RawTrackingData_Home_Team.csv": "0a3d0eff7785950379c587d3597b86a2d91c2492281b4652e1abb97c99b6b364",
    },
    2: {
        "Sample_Game_2_RawEventsData.csv": "edf31a18599265b77a8baf150f2ce6d89456fb0324b62ea7a391229657a619ba",
        "Sample_Game_2_RawTrackingData_Away_Team.csv": "aa639f40762eade46c389020fe4458266b5ba14027b1e96e22e75db62803718b",
        "Sample_Game_2_RawTrackingData_Home_Team.csv": "3f9990ac204b07133e4202441d6fcbb386897c7c7f81e1bc3ce4d35bc8644694",
    },
    3: {
        "Sample_Game_3_events.json": "28a53bfe9eaf32535362335553d517cdce0eabb588af6166009081753751f4e0",
        "Sample_Game_3_metadata.xml": "009c7dcfcb66192014917ca1559c158f612353e390eafb85f8214b06a1c7dec1",
        "Sample_Game_3_tracking.txt": "570e6ce82c2d93bc92723af331d93c31110bec1d6dc0be07c6d44ca3f981d658",
    },
}

IDSSE_DOWNLOAD = "https://ndownloader.figshare.com/files"
# match id -> ((matchinformation file id, md5), (positions file id, md5)), from the figshare API
IDSSE_FILES: dict[str, tuple[tuple[int, str], tuple[int, str]]] = {
    "J03WMX": (
        (51643475, "bc5d5b21dbcf36aa30f4cd0428309f6b"),
        (51643514, "e4273a2adf5dbd9d4ec42def2f01905c"),
    ),
    "J03WN1": (
        (51643472, "37b0f85588c0acfb995fe633c7ac2e0d"),
        (51643517, "bacb2f609c1d1b0bfc906f6b9d4109f2"),
    ),
    "J03WOH": (
        (51643478, "b9710781d400dd812f62b0a6e3d7bce8"),
        (51643520, "3efff25a86a61768e7d253b4bba137a3"),
    ),
    "J03WOY": (
        (51643481, "7b021e577fca917a63111d55cc29ae5b"),
        (51643523, "4e1063e0a0ae499920cf68ac1aa3ab4e"),
    ),
    "J03WPY": (
        (51643487, "65a4568e4f374af94d3c1015cbc7c4f5"),
        (51643526, "e56ff1336a334f8da51110fbe7cab764"),
    ),
    "J03WQQ": (
        (51643484, "5979209af316a914ce57347d0633df6c"),
        (51643529, "9105bad987191d3c3c362c1fce9a4fcd"),
    ),
    "J03WR9": (
        (51643490, "5e2d580d3d8ed97e32959de90d04992b"),
        (51643532, "e060bb7d7a5a3bfbd15bbf9bc631024d"),
    ),
}
HALF_GAP = 250  # empty rows between two DFL halves (ungraded), so they never touch

# Third source, for training only: SkillCorner open data (MIT, https://github.com/SkillCorner/
# opendata, pinned commit), 20 A-League 2024/25 matches of broadcast tracking at 10 fps with the
# team in possession; the ball is in play exactly inside the "phases of play".
SKILLCORNER_COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
SKILLCORNER_MATCHES = (
    1874553, 1886347, 1899585, 1925299, 1927964, 1953632, 1959846, 1986691, 1996435, 1996436,
    2006229, 2006363, 2007448, 2007721, 2010085, 2011166, 2013725, 2015213, 2016236, 2017461,
)  # fmt: skip
SKILLCORNER_FPS = 10
SKILLCORNER_FILL_S = 0.5  # a gap this short between two phases of play is not a stoppage

FloatArray = npt.NDArray[np.float64]


def all_games() -> list[str]:
    """The benchmark: every score is measured on these ten games."""
    return [f"metrica_{n}" for n in sorted(FILES)] + [f"dfl_{m}" for m in sorted(IDSSE_FILES)]


def extra_games() -> list[str]:
    """Training-only games (SkillCorner), never part of the benchmark scores."""
    return [f"sc_{m}" for m in SKILLCORNER_MATCHES]


@dataclass(frozen=True)
class Event:
    team: int | None  # H (0) = home, A (1) = away
    type: str  # upper case, e.g. "BALL OUT"
    subtypes: tuple[str, ...]  # upper case, e.g. ("HEAD", "ON TARGET", "GOAL")
    period: int
    start: int  # tracking frame id (1-based), as in the sample data
    end: int


@dataclass(frozen=True)
class ExternalGame:
    name: str
    first_frame_id: int  # frame id of row 0
    periods: tuple[tuple[int, int, int], ...]  # (period, first frame id, last frame id)
    ball: FloatArray  # (n, 2) normalised 0-1, NaN when not tracked
    players: FloatArray  # (n, p, 2) normalised, NaN when not on the pitch
    player_team: npt.NDArray[np.int8]  # (p,) H or A
    events: tuple[Event, ...]
    labels: Codes | None = None  # per-frame H/A/D/X when the source provides them (DFL)

    @property
    def n_frames(self) -> int:
        return int(self.ball.shape[0])


# -- download -------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(game: int, root: Path | None = None) -> Path:
    """Download one sample game (pinned commit), verify every sha256, return its folder."""
    folder = (root or cache_dir() / "external" / f"metrica-sample-data-{COMMIT[:10]}") / (
        f"Sample_Game_{game}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES[game].items():
        target = folder / name
        if not target.is_file() or _sha256(target) != expected:
            url = f"{UPSTREAM}/{COMMIT}/data/Sample_Game_{game}/{name}"
            tmp = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            if _sha256(tmp) != expected:
                tmp.unlink()
                raise ValueError(f"{name}: sha256 mismatch, refusing the download")
            tmp.replace(target)
    return folder


# -- parsing --------------------------------------------------------------------------------


def _subtypes(raw: Any) -> tuple[str, ...]:
    """'HEAD-ON TARGET-GOAL' (CSV), {'name': ...} or [{'name': ...}, ...] (JSON) -> tokens."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ()
    if isinstance(raw, str):
        return tuple(token.strip().upper() for token in raw.split("-") if token.strip())
    if isinstance(raw, dict):
        return (str(raw["name"]).upper(),)
    return tuple(str(item["name"]).upper() for item in raw)


def _tracking_csv(path: Path) -> tuple[pd.DataFrame, FloatArray, FloatArray]:
    """(index columns, players (n, p, 2), ball (n, 2)) from a Metrica CSV tracking file."""
    table = pd.read_csv(path, skiprows=2)
    values = table.to_numpy(dtype=np.float64)
    n_players = (values.shape[1] - 3 - 2) // 2
    players = values[:, 3 : 3 + 2 * n_players].reshape(len(table), n_players, 2)
    return table.iloc[:, :3], players, values[:, -2:]


def _load_csv_game(game: int, folder: Path) -> ExternalGame:
    index, home, ball = _tracking_csv(folder / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv")
    _, away, _ = _tracking_csv(folder / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv")
    frame_ids = index["Frame"].to_numpy(dtype=np.int64)
    if not np.array_equal(frame_ids, np.arange(frame_ids[0], frame_ids[0] + len(frame_ids))):
        raise ValueError(f"game {game}: tracking frames are not contiguous")
    periods = tuple(
        (int(p), int(rows["Frame"].min()), int(rows["Frame"].max()))
        for p, rows in index.groupby("Period")
    )
    raw = pd.read_csv(folder / f"Sample_Game_{game}_RawEventsData.csv")
    team_of = {"HOME": H, "AWAY": A}
    events = tuple(
        Event(
            team=team_of.get(str(row["Team"]).strip().upper()),
            type=str(row["Type"]).strip().upper(),
            subtypes=_subtypes(row["Subtype"]),
            period=int(row["Period"]),
            start=int(row["Start Frame"]),
            end=int(row["End Frame"]),
        )
        for _, row in raw.iterrows()
    )
    players = np.concatenate([home, away], axis=1)
    team = np.array([H] * home.shape[1] + [A] * away.shape[1], dtype=np.int8)
    return ExternalGame(f"metrica_{game}", int(frame_ids[0]), periods, ball, players, team, events)


def _load_epts_game(game: int, folder: Path) -> ExternalGame:
    """FIFA/EPTS: the order of the player groups on a line changes with substitutions, so each
    <DataFormatSpecification> block gets its own mapping from group to player."""
    root = ET.parse(folder / f"Sample_Game_{game}_metadata.xml").getroot()
    teams = [team.get("id") for team in root.iter("Team")]
    score = root.find(".//Score")
    home_id = (score.get("idLocalTeam") if score is not None else None) or teams[0]
    player_ids = [player.get("id") for player in root.iter("Player")]
    player_team = np.array(
        [H if player.get("teamId") == home_id else A for player in root.iter("Player")],
        dtype=np.int8,
    )
    column = {pid: i for i, pid in enumerate(player_ids)}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    blocks = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [ref.get("playerChannelId") for ref in spec.iter("PlayerChannelRef")]
        order = [column[channel_player[ref]] for ref in refs[::2]]  # x, y per player
        blocks.append((int(spec.get("startFrame", 0)), int(spec.get("endFrame", 0)), order))
    periods = []
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    for number, (first, last) in enumerate(
        (("first_half_start", "first_half_end"), ("second_half_start", "second_half_end")), 1
    ):
        if params.get(first) and params.get(last):
            periods.append((number, int(params[first] or 0), int(params[last] or 0)))

    lines = (folder / f"Sample_Game_{game}_tracking.txt").read_text(encoding="utf-8").splitlines()
    frame_ids = np.array([int(line[: line.index(":")]) for line in lines], dtype=np.int64)
    first_id = int(frame_ids.min())
    n = int(frame_ids.max()) - first_id + 1
    players = np.full((n, len(player_ids), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    for line, frame_id in zip(lines, frame_ids, strict=True):
        _, groups, ball_text = line.split(":")
        order = next(o for start, stop, o in blocks if start <= frame_id <= stop)
        row = frame_id - first_id
        for slot, group in zip(order, groups.split(";"), strict=False):
            x, y = group.split(",")[:2]
            players[row, slot] = (float(x), float(y))
        bx, by = ball_text.split(",")[:2]
        ball[row] = (float(bx), float(by))

    data = json.loads((folder / f"Sample_Game_{game}_events.json").read_text(encoding="utf-8"))
    events = tuple(
        Event(
            team=None if not item.get("team") else (H if item["team"]["id"] == home_id else A),
            type=str(item["type"]["name"]).upper(),
            subtypes=_subtypes(item.get("subtypes")),
            period=int(item["period"]),
            start=int(item["start"]["frame"]),
            end=int(item["end"]["frame"]),
        )
        for item in data["data"]
    )
    return ExternalGame(
        f"metrica_{game}", first_id, tuple(periods), ball, players, player_team, events
    )


# -- IDSSE / DFL -------------------------------------------------------------------------------


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(file_id: int, md5: str, target: Path) -> Path:
    if target.is_file() and _md5(target) == md5:
        return target
    tmp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(
        f"{IDSSE_DOWNLOAD}/{file_id}", headers={"User-Agent": "mtah-bench"}
    )
    with urllib.request.urlopen(request) as response, open(tmp, "wb") as handle:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            handle.write(chunk)
    if _md5(tmp) != md5:
        tmp.unlink()
        raise ValueError(f"figshare file {file_id}: md5 mismatch, refusing the download")
    tmp.replace(target)
    return target


def _parse_dfl(
    info_path: Path, positions_path: Path
) -> tuple[
    FloatArray, FloatArray, npt.NDArray[np.int8], Codes, list[tuple[int, int]], tuple[float, float]
]:
    """(ball (n, 2) normalised, players (n, p, 2) normalised, player_team, labels, half row spans,
    pitch) from DFL XML. Streams the ~400 MB position file; referees are dropped."""
    teams = ET.parse(info_path).getroot().iter("Team")
    side_of = {team.get("TeamId"): (H if team.get("Role") == "home" else A) for team in teams}
    pitch = PITCH
    per_section: dict[str, dict[str, Any]] = {}
    current: tuple[str, str, str] | None = None
    rows: list[tuple[float, ...]] = []
    for event, elem in ET.iterparse(positions_path, events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            current = (
                elem.get("GameSection", ""),
                elem.get("TeamId", ""),
                elem.get("PersonId", ""),
            )
            rows = []
        elif event == "end" and elem.tag == "Frame" and current is not None:
            if current[1].upper() == "BALL":
                rows.append(
                    (
                        float(elem.get("N", "nan")),
                        float(elem.get("X", "nan")),
                        float(elem.get("Y", "nan")),
                        float(elem.get("BallStatus", "nan")),
                        float(elem.get("BallPossession", "nan")),
                    )
                )
            else:
                rows.append(
                    (
                        float(elem.get("N", "nan")),
                        float(elem.get("X", "nan")),
                        float(elem.get("Y", "nan")),
                    )
                )
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet" and current is not None:
            section, team, person = current
            if team.lower() != "referee" and rows:
                store = per_section.setdefault(section, {"players": {}, "ball": None})
                if team.upper() == "BALL":
                    store["ball"] = np.array(rows)
                elif team in side_of:
                    store["players"][person] = (side_of[team], np.array(rows))
            current = None
            elem.clear()
        elif event == "end" and elem.tag == "PitchSize":
            pitch = (float(elem.get("X", PITCH[0])), float(elem.get("Y", PITCH[1])))
    people = sorted({p for s in per_section.values() for p in s["players"]})
    column = {p: i for i, p in enumerate(people)}
    player_side = np.zeros(len(people), dtype=np.int8)
    blocks, spans, start = [], [], 0
    for section in sorted(per_section, key=lambda s: per_section[s]["ball"][0, 0]):
        store = per_section[section]
        ball_rows = store["ball"]
        first_n, last_n = int(ball_rows[:, 0].min()), int(ball_rows[:, 0].max())
        n = last_n - first_n + 1
        ball = np.full((n, 2), np.nan)
        status = np.full(n, np.nan)
        possession = np.full(n, np.nan)
        idx = ball_rows[:, 0].astype(np.int64) - first_n
        ball[idx] = ball_rows[:, 1:3]
        status[idx], possession[idx] = ball_rows[:, 3], ball_rows[:, 4]
        players = np.full((n, len(people), 2), np.nan)
        for person, (side, person_rows) in store["players"].items():
            keep = (person_rows[:, 0] >= first_n) & (person_rows[:, 0] <= last_n)
            players[person_rows[keep, 0].astype(np.int64) - first_n, column[person]] = person_rows[
                keep, 1:3
            ]
            player_side[column[person]] = side
        labels = np.full(n, X, dtype=np.int8)
        labels[status == 0] = D
        labels[(status == 1) & (possession == 1)] = H
        labels[(status == 1) & (possession == 2)] = A
        blocks.append((ball, players, labels))
        spans.append((start, start + n - 1))
        start += n + HALF_GAP
    total = start - HALF_GAP
    ball_all = np.full((total, 2), np.nan)
    players_all = np.full((total, len(people), 2), np.nan)
    labels_all = np.full(total, X, dtype=np.int8)
    for (ball, players, labels), (a, b) in zip(blocks, spans, strict=True):
        ball_all[a : b + 1], players_all[a : b + 1], labels_all[a : b + 1] = ball, players, labels
    scale = np.asarray(pitch)
    # DFL coordinates are metres from the centre spot; ours are 0-1 from a corner.
    return (
        ball_all / scale + 0.5,
        players_all / scale + 0.5,
        player_side,
        labels_all,
        spans,
        pitch,
    )


def load_dfl_game(match_id: str, root: Path | None = None, keep_raw: bool = False) -> ExternalGame:
    """One IDSSE match. The ~400 MB XML is parsed once into a compact cache, then deleted
    (unless keep_raw): the disk only ever holds one raw file at a time."""
    folder = (root or cache_dir() / "external" / "idsse") / match_id
    folder.mkdir(parents=True, exist_ok=True)
    compact = folder / "clean.npz"
    if not compact.is_file():
        (info_id, info_md5), (pos_id, pos_md5) = IDSSE_FILES[match_id]
        info = _download(info_id, info_md5, folder / "matchinformation.xml")
        positions = _download(pos_id, pos_md5, folder / "positions.xml")
        ball, players, team, labels, spans, pitch = _parse_dfl(info, positions)
        np.savez_compressed(
            compact,
            ball=ball.astype(np.float32),
            players=players.astype(np.float32),
            team=team,
            labels=labels,
            spans=np.array(spans),
            pitch=np.array(pitch),
        )
        if not keep_raw:
            positions.unlink()
    with np.load(compact) as data:
        spans = [(int(a), int(b)) for a, b in data["spans"]]
        return ExternalGame(
            name=f"dfl_{match_id}",
            first_frame_id=1,
            periods=tuple((i + 1, a + 1, b + 1) for i, (a, b) in enumerate(spans)),
            ball=data["ball"].astype(np.float64),
            players=data["players"].astype(np.float64),
            player_team=data["team"].astype(np.int8),
            events=(),
            labels=data["labels"].astype(np.int8),
        )


def _fetch_skillcorner(match_id: int, folder: Path) -> tuple[Path, Path, Path]:
    """match.json, phases of play and tracking (git LFS, sha256 from its pointer) at the pinned
    commit."""
    raw = f"https://raw.githubusercontent.com/SkillCorner/opendata/{SKILLCORNER_COMMIT}"
    media = f"https://media.githubusercontent.com/media/SkillCorner/opendata/{SKILLCORNER_COMMIT}"
    stem = f"data/matches/{match_id}/{match_id}"

    def get(url: str, target: Path) -> Path:
        if not target.is_file():
            request = urllib.request.Request(url, headers={"User-Agent": "mtah-bench"})
            tmp = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(request) as response, open(tmp, "wb") as handle:
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    handle.write(chunk)
            tmp.replace(target)
        return target

    match = get(f"{raw}/{stem}_match.json", folder / "match.json")
    phases = get(f"{raw}/{stem}_phases_of_play.csv", folder / "phases_of_play.csv")
    pointer = get(f"{raw}/{stem}_tracking_extrapolated.jsonl", folder / "tracking.pointer")
    oid = next(
        line.split(":", 1)[1].strip()
        for line in pointer.read_text().splitlines()
        if line.startswith("oid sha256:")
    )
    tracking = get(f"{media}/{stem}_tracking_extrapolated.jsonl", folder / "tracking.jsonl")
    if _sha256(tracking) != oid:
        tracking.unlink()
        raise ValueError(f"SkillCorner {match_id}: tracking sha256 mismatch, refusing the download")
    return match, phases, tracking


def _parse_skillcorner(
    match_path: Path, phases_path: Path, tracking_path: Path
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.int8], Codes, list[tuple[int, int]]]:
    """(ball, players, player_team, labels, half row spans) at 25 fps, normalised 0-1.

    Players: SkillCorner's positions, extrapolated off camera (NaN off the pitch). Ball: only
    where detected. Label: `D` outside the phases of play (defined only while the ball is in
    play), else the phase's team in possession; a gap shorter than SKILLCORNER_FILL_S between two
    phases keeps the previous team."""
    match = json.loads(match_path.read_text(encoding="utf-8"))
    pitch = np.array([float(match["pitch_length"]), float(match["pitch_width"])])
    home = match["home_team"]["id"]
    people = sorted(p["id"] for p in match["players"])
    column = {p: i for i, p in enumerate(people)}
    side = {p["id"]: (H if p["team_id"] == home else A) for p in match["players"]}
    player_team = np.array([side[p] for p in people], dtype=np.int8)
    frames: dict[int, list[tuple[int, Any]]] = {1: [], 2: []}
    with open(tracking_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("period") in frames:
                frames[row["period"]].append((int(row["frame"]), row))
    phases = pd.read_csv(phases_path)
    team_of = {home: H, match["away_team"]["id"]: A}
    blocks, spans, start = [], [], 0
    for period in (1, 2):
        rows = sorted(frames[period], key=lambda item: item[0])
        first, last = rows[0][0], rows[-1][0]
        n10 = last - first + 1
        ball10 = np.full((n10, 2), np.nan)
        players10 = np.full((n10, len(people), 2), np.nan)
        for frame, row in rows:
            ball = row.get("ball_data") or {}
            if ball.get("is_detected") and ball.get("x") is not None:
                ball10[frame - first] = (ball["x"], ball["y"])
            for p in row.get("player_data") or []:
                if p.get("player_id") in column and p.get("x") is not None:
                    players10[frame - first, column[p["player_id"]]] = (p["x"], p["y"])
        labels10 = np.full(n10, D, dtype=np.int8)
        in_period = phases[phases["period"] == period].sort_values("frame_start")
        previous_end, previous_team = -(10**9), D  # nothing to bridge before the first phase
        for phase in in_period.itertuples():
            a, b = int(phase.frame_start) - first, int(phase.frame_end) - first
            team = team_of.get(int(phase.team_in_possession_id))
            if team is None:
                continue
            if 0 < a - previous_end <= SKILLCORNER_FILL_S * SKILLCORNER_FPS:
                labels10[max(0, previous_end) : max(0, a)] = previous_team
            labels10[max(0, a) : max(0, b + 1)] = team
            previous_end, previous_team = b + 1, team
        # 10 -> 25 fps: positions interpolated, label of the nearest original frame
        n = int((n10 - 1) * FPS / SKILLCORNER_FPS) + 1
        source = np.arange(n) * SKILLCORNER_FPS / FPS
        lo = np.floor(source).astype(np.int64)
        hi = np.minimum(lo + 1, n10 - 1)
        w = (source - lo)[:, None]
        ball = ball10[lo] * (1 - w) + ball10[hi] * w
        players = players10[lo] * (1 - w[:, :, None]) + players10[hi] * w[:, :, None]
        labels = labels10[np.minimum(np.rint(source).astype(np.int64), n10 - 1)]
        blocks.append((ball, players, labels))
        spans.append((start, start + n - 1))
        start += n + HALF_GAP
    total = start - HALF_GAP
    ball_all = np.full((total, 2), np.nan)
    players_all = np.full((total, len(people), 2), np.nan)
    labels_all = np.full(total, X, dtype=np.int8)
    for (ball, players, labels), (a, b) in zip(blocks, spans, strict=True):
        ball_all[a : b + 1], players_all[a : b + 1], labels_all[a : b + 1] = ball, players, labels
    # metres from the centre spot -> 0-1 from a corner
    return ball_all / pitch + 0.5, players_all / pitch + 0.5, player_team, labels_all, spans


def load_skillcorner_game(match_id: int, root: Path | None = None) -> ExternalGame:
    """One SkillCorner match, parsed once into a compact cache; the ~90 MB tracking is deleted."""
    folder = (root or cache_dir() / "external" / "skillcorner") / str(match_id)
    folder.mkdir(parents=True, exist_ok=True)
    compact = folder / "clean.npz"
    if not compact.is_file():
        match, phases, tracking = _fetch_skillcorner(match_id, folder)
        ball, players, team, labels, spans = _parse_skillcorner(match, phases, tracking)
        np.savez_compressed(
            compact,
            ball=ball.astype(np.float32),
            players=players.astype(np.float32),
            team=team,
            labels=labels,
            spans=np.array(spans),
        )
        tracking.unlink()
    with np.load(compact) as data:
        spans = [(int(a), int(b)) for a, b in data["spans"]]
        return ExternalGame(
            name=f"sc_{match_id}",
            first_frame_id=1,
            periods=tuple((i + 1, a + 1, b + 1) for i, (a, b) in enumerate(spans)),
            ball=data["ball"].astype(np.float64),
            players=data["players"].astype(np.float64),
            player_team=data["team"].astype(np.int8),
            events=(),
            labels=data["labels"].astype(np.int8),
        )


def load_game(game: int | str, root: Path | None = None) -> ExternalGame:
    """`metrica_1`..`metrica_3` (or 1..3), `dfl_<match id>` or `sc_<match id>`."""
    name = f"metrica_{game}" if isinstance(game, int) else game
    if name.startswith("dfl_"):
        return load_dfl_game(name.removeprefix("dfl_"), root)
    if name.startswith("sc_"):
        return load_skillcorner_game(int(name.removeprefix("sc_")), root)
    number = int(name.removeprefix("metrica_"))
    folder = fetch(number, root)
    return _load_epts_game(number, folder) if number == 3 else _load_csv_game(number, folder)


# -- ground truth from the events ----------------------------------------------------------

POSSESSION_TYPES = frozenset(
    {"PASS", "CARRY", "RECOVERY", "SHOT", "BALL LOST", "BALL OUT", "SET PIECE"}
)


def is_restart(event: Event) -> bool:
    """A set piece, or a goal kick (Metrica tags those as PASS / BALL LOST with GOAL KICK)."""
    return event.type == "SET PIECE" or "GOAL KICK" in event.subtypes


def stoppage_frame(event: Event) -> int | None:
    """Frame id at which this event stops play, or None if it does not."""
    tags = set(event.subtypes)
    if event.type == "BALL OUT":
        return event.end  # the ball crosses the line at the end of the event
    if event.type == "SHOT" and tags & {"GOAL", "OUT"}:
        return event.end
    if "ADVANTAGE" in tags:
        return None  # play goes on
    if event.type == "FAULT RECEIVED" or (event.type == "CHALLENGE" and "FAULT" in tags):
        return event.start
    if event.type == "BALL LOST" and tags & {"OFFSIDE", "HAND BALL"}:
        return event.end
    if event.type == "CARD":
        return event.start  # shown during a stoppage
    return None


def derive_labels(game: ExternalGame) -> Codes:
    """Frame-level H/A/D (X outside the periods), mechanically from the human event data.

    Alive: the team of the latest possession event (pass, carry, recovery, shot, ball lost, ball
    out, set piece): possession changes when the other team recovers the ball, and stays with
    the last team through loose balls and duels (CHALLENGE events change nothing on their own).
    Dead: from the stoppage that precedes a restart (ball out, goal, foul, offside, hand ball,
    card; failing that, the end of the last event) to the frame before the restart, and from the
    start of a period to its kick-off.
    """
    n, first = game.n_frames, game.first_frame_id
    codes = np.full(n, X, dtype=np.int8)
    for period, first_id, last_id in game.periods:
        lo, hi = first_id - first, last_id - first + 1  # rows of this period, half-open
        events = sorted(
            (e for e in game.events if e.period == period and first_id <= e.start <= last_id),
            key=lambda e: e.start,
        )
        team = np.full(hi - lo, -1, dtype=np.int8)
        for event in events:
            if event.type in POSSESSION_TYPES and event.team is not None:
                team[event.start - first_id :] = event.team
        dead = team < 0  # before the first possession event of the period
        previous_restart = first_id
        for index, event in enumerate(events):
            if not is_restart(event):
                continue
            window = [e for e in events[:index] if e.start >= previous_restart]
            stops = [f for f in (stoppage_frame(e) for e in window) if f is not None]
            stops = [f for f in stops if f <= event.start]
            if stops:
                stop = max(stops)
            elif window:
                stop = max(min(e.end, event.start) for e in window)
            else:
                stop = first_id  # the period's kick-off
            dead[max(0, stop - first_id) : event.start - first_id] = True
            previous_restart = event.start
        codes[lo:hi] = np.where(dead, D, team)
    return codes


# -- degradation towards the automatic ATD -------------------------------------------------


@dataclass(frozen=True)
class Degradation:
    """How far the clean tracking is pushed towards the challenge's automatic tracking.

    Published facts about BAR-ATH (DATA.md): 46 slots for 22 players, slots appear, vanish and
    swap identity, 20.5 tracked objects on average during play, more than 22 when the tracker
    duplicates someone, and the ball missing on 41 % of graded frames, "disproportionately" in
    crowded boxes, aerial duels and tackles. The magnitudes below are chosen to reproduce those
    published figures; nothing in them comes from the BAR-ATH data itself.

    The ball needs the most care. These sample games hide it during stoppages (it is missing on
    85-99 % of dead frames and 3-7 % of live ones), which would turn "no ball" into a free
    dead-ball detector that does not exist in BAR-ATH. So dead stretches get a plausible ball
    path first, then one dropout process runs over the whole match.
    """

    mean_tracklet_s: float = 25.0  # a slot follows one player this long on average
    mean_gap_s: float = 1.8  # then loses him this long (=> ~93 % of players tracked)
    duplicate_rate_per_min: float = 0.6  # a second slot follows the same player briefly
    position_noise_m: float = 0.15
    ball_noise_m: float = 0.2
    ball_missing_live: float = 0.38  # target share of live frames without a ball
    ball_missing_dead: float = 0.45  # ... and of dead frames
    ball_mean_gap_s: float = 1.5  # a dropout lasts this long on average
    ball_lost_stoppage: float = 0.2  # share of stoppages where the ball is never seen
    crowd_radius_m: float = 2.0  # >= 2 players this close: occlusion more likely
    fast_ball_ms: float = 15.0  # a ball this fast (in the air, a shot) is lost more often
    blackout_share: float = 0.0  # share of the time with a large part of the players untracked
    dead_ball_motion: float = 0.0  # share of the stoppages whose ball keeps moving (deep_data)


# The same games as an automatic tracking of this kind really looks, from a label-free description
# of the provided match's tracking in coarse bins (docs/TECHNICAL_DOCUMENT.md): smooth positions
# (jitter under a centimetre), a slot follows a player for about two minutes, the ball is seen or
# missing in runs of tens of seconds, and about a tenth of the time many players are not tracked.
# Nothing here encodes a label or a frame of that match.
REAL_LIKE = Degradation(
    mean_tracklet_s=120.0,
    mean_gap_s=3.0,
    duplicate_rate_per_min=0.3,
    position_noise_m=0.003,
    ball_noise_m=0.005,
    ball_missing_live=0.25,  # with the stoppages below: about 60 % of the frames have a ball
    ball_missing_dead=0.60,  # a ball lying still is rarely tracked (seen but still: 5-15 %)
    ball_mean_gap_s=20.0,
    blackout_share=0.10,
)
# ... and with the ball of most stoppages moving and seen, as an automatic tracking shows it
# (seen but still: 5-15 % of the frames with a ball, against more than 30 % in the sources).
REAL_LIKE_MOVING = dataclasses.replace(
    REAL_LIKE, dead_ball_motion=0.7, ball_missing_live=0.38, ball_missing_dead=0.30
)
PROFILES = {
    "simulated": Degradation(),
    "real-like": REAL_LIKE,
    "real-like-moving": REAL_LIKE_MOVING,
}
_PROFILE_DIRS = {
    "simulated": "bench",
    "real-like": "bench_real",
    "real-like-moving": "bench_real_moving",
}


def _gap_rows(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True)
    )


def _dead_ball_path(
    ball: FloatArray, start: int, stop: int, rng: np.random.Generator, settings: Degradation
) -> None:
    """Fill a dead stretch [start, stop) in place: never seen at all (lost from view), the source's
    own ball when it tracked it through the stoppage (DFL), or else a plausible path from where the
    ball left play to the restart spot, then still there (placed ball)."""
    if rng.random() < settings.ball_lost_stoppage:
        ball[start:stop] = np.nan
        return
    if np.mean(~np.isnan(ball[start:stop, 0])) > 0.5:
        return
    before = ball[max(0, start - 50) : start]
    after = ball[stop : stop + 50]
    known_before = before[~np.isnan(before[:, 0])]
    known_after = after[~np.isnan(after[:, 0])]
    if not len(known_before) or not len(known_after):
        ball[start:stop] = np.nan
        return
    p0, p1 = known_before[-1], known_after[0]
    n = stop - start
    travel = min(n, int(rng.uniform(0.3, 0.8) * n) + 1)  # fetched, then placed or held
    w = np.linspace(0.0, 1.0, travel)[:, None]
    ball[start : start + travel] = p0 + w * (p1 - p0)
    ball[start + travel : stop] = p1


def _ball_dropouts(
    ball: FloatArray,
    players: FloatArray,
    dead: npt.NDArray[np.bool_],
    rng: np.random.Generator,
    settings: Degradation,
) -> npt.NDArray[np.bool_]:
    """Missing-ball mask from a two-state chain whose on->off hazard rises when the ball is
    crowded or fast; scaled so that live and dead frames reach their target missing shares."""
    scale = np.asarray(PITCH)
    ball_m = ball * scale
    near = np.linalg.norm(players * scale - ball_m[:, None, :], axis=2)
    crowded = np.nansum(near < settings.crowd_radius_m, axis=1) >= 2
    speed = np.full(len(ball), 0.0)
    speed[1:] = np.nan_to_num(np.linalg.norm(np.diff(ball_m, axis=0), axis=1) * FPS)
    fast = speed > settings.fast_ball_ms
    modulation = 1.0 + 2.0 * crowded + 2.0 * fast
    recover = 1.0 / (settings.ball_mean_gap_s * FPS)
    hazard = np.empty(len(ball))
    for state, target in ((False, settings.ball_missing_live), (True, settings.ball_missing_dead)):
        rows = dead == state
        if rows.any():
            base = target * recover / (1.0 - target)
            hazard[rows] = base * modulation[rows] / modulation[rows].mean()
    draws = rng.random(len(ball))
    missing = np.zeros(len(ball), dtype=bool)
    off = False
    for i in range(len(ball)):
        off = draws[i] >= recover if off else draws[i] < hazard[i]
        missing[i] = off
    return missing


def _renewal(
    rng: np.random.Generator, n: int, mean_on: float, mean_off: float
) -> list[tuple[int, int]]:
    """[start, stop) runs of an alternating on/off process (exponential durations, frames)."""
    runs, t, on = [], 0, bool(rng.random() < mean_on / (mean_on + mean_off))
    while t < n:
        length = max(1, int(rng.exponential(mean_on if on else mean_off)))
        if on:
            runs.append((t, min(n, t + length)))
        t += length
        on = not on
    return runs


def degrade(
    game: ExternalGame, codes: Codes, seed: int = 0, settings: Degradation | None = None
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.int8]]:
    """(slots (n, 46, 2), ball (n, 2), slot_team (46,)): anonymous, fragmented, noisy tracking.

    `codes` (the derived labels) only says where the stoppages are, so that the ball can be given
    a path there; the degraded tracking never encodes the labels themselves.
    """
    s = settings or Degradation()
    rng = np.random.default_rng(seed)
    n = game.n_frames
    scale = np.asarray(PITCH)
    slot_team = np.array([H] * (N_SLOTS // 2) + [A] * (N_SLOTS // 2), dtype=np.int8)
    rng.shuffle(slot_team)  # interleaved, as the real metadata may be
    slots = np.full((n, N_SLOTS, 2), np.nan)
    free_at = np.zeros(N_SLOTS, dtype=np.int64)  # first frame each slot is free again
    tracklets: list[tuple[int, int, int]] = []  # (start, stop, player)
    for player in range(game.players.shape[1]):
        on_pitch = ~np.isnan(game.players[:, player, 0])
        if not on_pitch.any():
            continue
        for start, stop in _renewal(rng, n, s.mean_tracklet_s * FPS, s.mean_gap_s * FPS):
            rows = np.flatnonzero(on_pitch[start:stop]) + start
            if len(rows):
                tracklets.append((int(rows[0]), int(rows[-1]) + 1, player))
    tracklets.sort()
    for start, stop, player in tracklets:
        team_slots = np.flatnonzero((slot_team == game.player_team[player]) & (free_at <= start))
        if not len(team_slots):
            continue  # every slot of that team is busy: the tracker misses him
        slot = int(rng.choice(team_slots))
        free_at[slot] = stop
        slots[start:stop, slot] = game.players[start:stop, player]
    # Duplicates: another free slot of the same team shadows a player for 1-3 s.
    n_dup = int(rng.poisson(s.duplicate_rate_per_min * n / FPS / 60))
    for _ in range(n_dup):
        player = int(rng.integers(game.players.shape[1]))
        start = int(rng.integers(n))
        stop = min(n, start + int(rng.uniform(1, 3) * FPS))
        team_slots = np.flatnonzero((slot_team == game.player_team[player]) & (free_at <= start))
        if not len(team_slots) or np.isnan(game.players[start, player, 0]):
            continue
        slot = int(rng.choice(team_slots))
        free_at[slot] = stop
        offset = rng.normal(0.0, 0.7, 2) / scale
        slots[start:stop, slot] = game.players[start:stop, player] + offset
    slots += rng.normal(0.0, s.position_noise_m, slots.shape) / scale
    if s.blackout_share > 0:
        from mtah.solutions.ali.deep_data import blackouts  # one definition for both

        blackouts(slots, game.ball[:, 0].astype(np.float64), rng, s.blackout_share, (2.0, 14.0))

    ball = game.ball.copy()
    dead = codes == D
    for start, stop in _gap_rows(dead):
        _dead_ball_path(ball, start, stop, rng, s)
    if s.dead_ball_motion > 0:
        from mtah.solutions.ali.deep_data import dead_ball_motion

        dead_ball_motion(ball, game.players.astype(np.float64), dead, rng, s.dead_ball_motion)
    missing = _ball_dropouts(ball, game.players, dead, rng, s)
    ball[missing] = np.nan
    ball += rng.normal(0.0, s.ball_noise_m, ball.shape) / scale
    return slots, ball, slot_team


def slot_speed(slots: FloatArray) -> FloatArray:
    """(n, 46) speed in m/s over +/- 2 frames, NaN at the ends of a tracklet."""
    metres = slots * np.asarray(PITCH)
    speed = np.full(slots.shape[:2], np.nan)
    speed[2:-2] = np.linalg.norm(metres[4:] - metres[:-4], axis=2) * FPS / 4.0
    return speed


# -- writing a bundle ------------------------------------------------------------------------


def _write_atd(path: Path, slots: FloatArray, speed: FloatArray, ball: FloatArray) -> None:
    n = slots.shape[0]
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for start in range(0, n, 20_000):
            rows = []
            for i in range(start, min(n, start + 20_000)):
                groups = [
                    "NaN,NaN,NaN"
                    if np.isnan(x)
                    else f"{x:.5f},{y:.5f},{0.0 if np.isnan(v) else v:.2f}"
                    for (x, y), v in zip(slots[i], speed[i], strict=True)
                ]
                bx, by = ball[i]
                ball_text = "NaN,NaN,NaN" if np.isnan(bx) else f"{bx:.5f},{by:.5f},0.00"
                rows.append(f"{i + 1}:{';'.join(groups)}:{ball_text}\n")
            handle.write("".join(rows))


def _write_metadata(path: Path, slot_team: npt.NDArray[np.int8], source: str) -> None:
    ids = ("TEAMHOME", "TEAMAWAY")
    players = "\n".join(
        f'      <Player id="T{i + 1}" teamId="{ids[int(t)]}"><Name>Track {i + 1}</Name>'
        f"<ShirtNumber>{i + 1}</ShirtNumber></Player>"
        for i, t in enumerate(slot_team)
    )
    path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<main>
  <Metadata>
    <GlobalConfig>
      <FrameRate>{FPS}</FrameRate>
      <ProviderName>{source}</ProviderName>
    </GlobalConfig>
    <Teams>
      <Team id="{ids[0]}"><Name>Home ({source})</Name></Team>
      <Team id="{ids[1]}"><Name>Away ({source})</Name></Team>
    </Teams>
    <Players>
{players}
    </Players>
  </Metadata>
</main>
""",
        encoding="utf-8",
        newline="\n",
    )


def _write_labels(path: Path, codes: Codes) -> None:
    letters = np.array(["H", "A", "D", "X"])[codes]
    rows = np.char.add(np.char.add(np.arange(len(codes)).astype(str), ","), letters)
    path.write_text(
        "frame,label\n" + "\n".join(rows.tolist()) + "\n", encoding="utf-8", newline="\n"
    )


def bundle_spec(game: ExternalGame) -> MatchSpec:
    """Halves in rows; the last frame of a period is left ungraded so that two adjacent periods
    never merge into one run in the official scorer (it splits runs on ground-truth X)."""
    first = game.first_frame_id
    halves = tuple(
        (period, a - first, b - first - (1 if i < len(game.periods) - 1 else 0))
        for i, (period, a, b) in enumerate(game.periods)
    )
    return MatchSpec("BAR-ATH", game.n_frames, FPS, PITCH, halves, synthetic=True)


def write_bundle(
    game: ExternalGame, out: Path, seed: int = 0, profile: str = "simulated"
) -> dict[str, Any]:
    spec = bundle_spec(game)
    codes = game.labels.copy() if game.labels is not None else derive_labels(game)
    codes[~spec.graded_mask()] = X
    slots, ball, slot_team = degrade(game, codes, seed, PROFILES[profile])
    tactical, meta_dir = out / "tactical", out / "synthetic"
    tactical.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    dfl = game.name.startswith("dfl_")
    source = (
        f"IDSSE / DFL open data (CC-BY 4.0), {game.name}"
        if dfl
        else f"Metrica Sports sample data, {game.name}"
    )
    _write_atd(tactical / "BAR-ATH_tactical_atd.txt", slots, slot_speed(slots), ball)
    _write_metadata(tactical / "BAR-ATH_tactical_atd.xml", slot_team, source)
    public = np.full_like(codes, X)
    first = spec.halves[0][1]
    public[first : first + 5 * 60 * FPS] = codes[first : first + 5 * 60 * FPS]
    _write_labels(meta_dir / "gt_full.csv", codes)
    _write_labels(meta_dir / "gt_public.csv", public)
    graded = spec.graded_mask()
    meta = {
        "kind": "external-benchmark",
        "warning": "NOT BAR-ATH: another match from public open data."
        + (
            " Labels: BallStatus / BallPossession of the DFL tracking."
            if dfl
            else " Labels: derived from its human event data."
        ),
        "source": source,
        "upstream": "https://doi.org/10.6084/m9.figshare.28196177"
        if dfl
        else f"https://github.com/metrica-sports/sample-data/tree/{COMMIT}",
        "bench_version": BENCH_VERSION,
        "profile": profile,
        "seed": seed,
        "spec": spec.to_dict(),
        "stats": {
            "ball_visible_graded": round(float((~np.isnan(ball[graded, 0])).mean()), 4),
            "tracked_objects_graded": round(
                float((~np.isnan(slots[graded, :, 0])).sum(axis=1).mean()), 2
            ),
            "label_counts_graded": {
                k: int(np.sum(codes[graded] == c)) for k, c in (("H", H), ("A", A), ("D", D))
            },
        },
    }
    with open(meta_dir / "meta.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")
    return meta


def default_out(profile: str = "simulated") -> Path:
    return cache_dir() / _PROFILE_DIRS[profile]


app = typer.Typer(add_completion=False, help=__doc__.splitlines()[0] if __doc__ else None)


@app.command()
def main(
    games: Annotated[
        list[str] | None, typer.Option("--game", "-g", help="metrica_1 .. dfl_J03WMX")
    ] = None,
    dataset: Annotated[str, typer.Option(help="all | metrica | dfl")] = "all",
    out: Annotated[Path | None, typer.Option(help="default: <cache>/bench")] = None,
    seed: Annotated[int, typer.Option()] = 0,
    profile: Annotated[
        str,
        typer.Option(help="simulated | real-like | real-like-moving (one cache folder each)"),
    ] = "simulated",
) -> None:
    """Build the benchmark bundles (downloads the public data on first use)."""
    root = out or default_out(profile)
    names = games or [g for g in all_games() if dataset == "all" or g.startswith(dataset)]
    for name in names:
        meta = write_bundle(load_game(name), root / name, game_seed(name, seed), profile)
        typer.echo(f"{name}: {json.dumps(meta['stats'])}")


def game_seed(name: str, seed: int = 0) -> int:
    """A different, reproducible degradation for every game."""
    return seed + zlib.crc32(name.encode()) % 10_000


def iter_bundles(root: Path | None = None, dataset: str = "all") -> list[Path]:
    """Benchmark bundles under root, `metrica_*` and `dfl_*` (dataset: all | metrica | dfl)."""
    base = root or default_out()
    return sorted(
        p
        for p in base.glob("*_*")
        if (p / "synthetic" / "meta.json").is_file()
        and (dataset == "all" or p.name.startswith(dataset))
    )


def describe_labels(codes: Codes, spec: MatchSpec) -> dict[str, float]:
    """How many possessions, how long: the order of magnitude a solution should produce."""
    from mtah.common.segments import spells

    lengths = [b - a for a, b, _ in spells(codes, spec)]
    graded = codes[spec.graded_mask()]
    return {
        "n_possessions": len(lengths),
        "mean_possession_s": float(np.mean(lengths)) / FPS if lengths else 0.0,
        "dead_share": float(np.mean(graded == D)),
        "possessions_per_5min": len(lengths) / (len(graded) / FPS / 300.0),
    }


if __name__ == "__main__":  # pragma: no cover
    app()
