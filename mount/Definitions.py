"""This module serves as a configuration file for interacting with the CARLA simulator in the context of autonomous
driving research. It defines various parameters and settings related to the simulation environment, including
connection details, logging configuration, synchronous mode settings, scenario configurations, weather and anomaly
specifications, NPC (Non-Player Character) configurations, and data processing parameters."""

import logging
from collections import namedtuple

import numpy as np
from aenum import Enum, NoAlias
# --- COLORS -----------------------------------------------------------------------------------------------------------

Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class
        "id",  # An integer ID that is associated with this label.
        # 'category'    , # The name of the category that this label belongs to
        # 'categoryId'  , # The ID of this category. Used to create ground truth images
        # on category level.
        "agent_mode",
        # actor is currently in agent mode and is more prone for unexpected behavior due to turned off autopilot.
        # It is therefore recommended to ignore these instances in evaluation
        "color",  # The color of this label
    ],
)

LABELS = [
    #       name                 id     agent_mode    color
    Label(
        "unlabeled",
        0,
        False,
        (0, 0, 0),
    ),
    Label(
        "road",
        1,
        False,
        (128, 64, 128),
    ),
    Label(
        "sidewalk",
        2,
        False,
        (244, 35, 232),
    ),
    Label(
        "building",
        3,
        False,
        (70, 70, 70),
    ),
    Label(
        "wall",
        4,
        False,
        (102, 102, 156),
    ),
    Label(
        "fence",
        5,
        False,
        (190, 153, 153),
    ),
    Label(
        "pole",
        6,
        False,
        (153, 153, 153),
    ),
    Label(
        "traffic light",
        7,
        False,
        (250, 170, 30),
    ),
    Label(
        "traffic sign ",
        8,
        False,
        (220, 220, 0),
    ),
    Label(
        "vegetation",
        9,
        False,
        (107, 142, 35),
    ),
    Label(
        "terrain",
        10,
        False,
        (152, 251, 152),
    ),
    Label(
        "sky",
        11,
        False,
        (70, 130, 180),
    ),
    Label(
        "pedestrian",
        12,
        False,
        (220, 20, 60),
    ),
    Label(
        "rider",
        13,
        False,
        (255, 0, 0),
    ),
    Label(
        "Car",
        14,
        False,
        (0, 0, 142),
    ),
    Label(
        "truck",
        15,
        False,
        (0, 0, 70),
    ),
    Label(
        "bus",
        16,
        False,
        (0, 60, 100),
    ),
    Label(
        "train",
        17,
        False,
        (0, 80, 100),
    ),
    Label(
        "motorcycle",
        18,
        False,
        (0, 0, 230),
    ),
    Label(
        "bicycle",
        19,
        False,
        (119, 11, 32),
    ),
    Label(
        "static",
        20,
        False,
        (110, 190, 160),
    ),
    Label(
        "dynamic",
        21,
        False,
        (170, 120, 50),
    ),
    Label(
        "other",
        22,
        False,
        (55, 90, 80),
    ),
    Label(
        "water",
        23,
        False,
        (45, 60, 150),
    ),
    Label(
        "road line",
        24,
        False,
        (157, 234, 50),
    ),
    Label(
        "ground",
        25,
        False,
        (81, 0, 81),
    ),
    Label(
        "bridge",
        26,
        False,
        (150, 100, 100),
    ),
    Label(
        "rail track",
        27,
        False,
        (230, 150, 140),
    ),
    Label(
        "guard rail",
        28,
        False,
        (180, 165, 180),
    ),
    # # anomalies
    # Label('home', 29, False, (245, 29, 0), ),
    # Label('animal', 30, False, (245, 30, 0), ),
    # Label('nature', 31, False, (245, 31, 0), ),
    # Label('special', 32, False, (245, 32, 0), ),
    # Label('falling', 34, False, (245, 34, 0), ),
    # Label('airplane', 33, False, (245, 33, 0), ),
    # Label(
    #     "anomaly",
    #     100,
    #     False,
    #     (245, 0, 0),
    # ),
    # agents
    Label(
        "agent_pedestrian",
        112,
        True,
        (220, 20, 61),
    ),
    Label(
        "agent_car",
        114,
        True,
        (3, 0, 143),
    ),
    Label(
        "agent_truck",
        115,
        True,
        (0, 0, 71),
    ),
    Label(
        "agent_bus",
        116,
        True,
        (0, 60, 101),
    ),
    Label("agent_rider", 113, True, (255, 0, 1)),
    Label(
        "agent_motorcycle",
        118,
        True,
        (0, 0, 231),
    ),
    Label(
        "agent_bicycle",
        119,
        True,
        (119, 11, 33),
    ),
    # misc
    Label(
        "ego_vehicle",
        214,
        True,
        (0, 0, 1),
    ),
]

WHITE_COLOR = [255, 255, 255]

ANOMALY_COLOR = [245, 0, 0]

EGO_COLOR = [0, 0, 0]

# endregion
# ======================================================================================================================
