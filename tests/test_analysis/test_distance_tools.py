import mdtraj as md
import numpy as np
import pandas as pd
import pytest

from mdcrow.tools.base_tools.analysis_tools.distance_tools import DistanceToolsUtils


@pytest.fixture(scope="module")
def distanceUtils(get_registry):
    # Test for the distanceToolsUtils class
    reg = get_registry("raw", False)
    return DistanceToolsUtils(path_registry=reg)


@pytest.fixture(scope="module")
def xyz_coords():
    """3x3x3 cube of points"""
    points = np.zeros((1, 27, 3), dtype=int)

    # Fill the array with coordinates of points in a 3x3x3 cube
    index = 0
    for x in range(3):
        for y in range(3):
            for z in range(3):
                points[0, index] = [x, y, z]
                index += 1
    return points


@pytest.fixture(scope="module")
def dummy_traj(xyz_coords):
    new_topology = pd.DataFrame(
        {
            "serial": range(xyz_coords.shape[1]),
            "name": ["CA" for _ in range(xyz_coords.shape[1])],
            "resSeq": range(xyz_coords.shape[1]),
            "resName": ["A" for _ in range(xyz_coords.shape[1])],
            "element": ["VS" for _ in range(xyz_coords.shape[1])],
            "chainID": [0 for _ in range(xyz_coords.shape[1])],
            "segmentID": [1 for _ in range(xyz_coords.shape[1])],
        }
    )
    top = md.Topology.from_dataframe(new_topology)
    new_traj = md.Trajectory(xyz_coords, top)
    return new_traj


def test_distanceToolsUtils_init(get_registry):
    # Test the init method of the distanceToolsUtils class
    reg = get_registry("raw", False)
    distance_utils = DistanceToolsUtils(path_registry=reg)
    assert distance_utils.path_registry is not None


def test_distanceToolsUtils_all_possible_pairs(distanceUtils):
    # Test the all_possible_pairs method of the distanceToolsUtils class
    List_1 = [1, 2, 3]
    List_2 = [4, 5, 6]
    util = distanceUtils
    pairs = util.all_possible_pairs(List_1, List_2)
    assert pairs == [
        (1, 4),
        (1, 5),
        (1, 6),
        (2, 4),
        (2, 5),
        (2, 6),
        (3, 4),
        (3, 5),
        (3, 6),
    ]


def test_distanceToolsUtils_get_contact_matrix(distanceUtils, dummy_traj):
    # Test the get_contact_matrix method of the distanceToolsUtils class
    contact_matrix = distanceUtils.calc_matrix_cm_all_resids(
        dummy_traj, threshold=4, distance=5
    )
    assert contact_matrix.shape == (1, 27, 27)
    assert np.sum(contact_matrix) == 27 * 27  # every residue is in contact

    no_contact_matrix = distanceUtils.calc_matrix_cm_all_resids(
        dummy_traj, threshold=0.5, distance=0
    )
    print(no_contact_matrix)
    assert no_contact_matrix.shape == (1, 27, 27)
    assert np.sum(no_contact_matrix) == 129  # no residue is in contact, but the dist
    # between a residue and its i+2
    # neighbor and itself is 0.
