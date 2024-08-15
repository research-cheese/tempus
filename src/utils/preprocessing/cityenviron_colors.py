from utils.core.hex_to_rgb import hex_to_rgb

HUMAN_COLORS = [
    "bd87bc",
    # "e2958f",
]

VEHICLE_COLORS = [
    "6fb275"
]

CONSTRUCTION_COLORS = [
"3c8a60",
"38d61a",
"bfc390",
"38d61a",
"c6b47a",
"3b25d4",
# "6dce18",
"6c74e0",
"e0baf9",
"fba480",
"3c8a60",
"b891b6",
"c6b47a",
"6c74e0",
# "a51c3d",
# "f03bd2",
"d07d44",
"431123",
"f47533",
"bfc390",
"bfc390",
"193eae",
"f5166e",
# "510d24",
"163df7",
"60e82c",
"c41e08",
"e688c6",
"d1f7ca",
"3b25d4",
"1c226c",
"062e56",
# "b4bf1d",
"45942e",
"c3ed84",
"45942e",
"003541",
# "109a04"
]

SKY_COLORS = ["a4c211"]

GROUND_COLORS = [
"bb469c",
"52efe8",
"aab32a",
"25807d",
"52efe8",
"7069bf",
"aef9c5"
]

NATURE_COLORS = [
"63f268",
"977eab",
"1aa5a6",
"cd78a1",
"1aa5a6",
"c22707",
"cd78a1",
# "e2958f",
"d4333c",
"cd78a1"
]

HUMAN_COLORS = set([hex_to_rgb(c) for c in HUMAN_COLORS])
VEHICLE_COLORS = set([hex_to_rgb(c) for c in VEHICLE_COLORS])
CONSTRUCTION_COLORS = set([hex_to_rgb(c) for c in CONSTRUCTION_COLORS])
SKY_COLORS = set([hex_to_rgb(c) for c in SKY_COLORS])
GROUND_COLORS = set([hex_to_rgb(c) for c in GROUND_COLORS])
NATURE_COLORS = set([hex_to_rgb(c) for c in NATURE_COLORS])