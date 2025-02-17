import pytest
from openmm import unit
from openmm.app import PME, NoCutoff

from mdcrow.tools.base_tools.simulation_tools.setup_and_run import OpenMMSimulation


@pytest.fixture
def setup_run_input():
    # random values
    return {
        "pdb_id": "1ABC",
        "forcefield_files": ["amber14-all.xml", "amber14/tip3p.xml"],
        "save": True,
        "system_params": {
            "nonbondedMethod": "PME",
            "nonbondedCutoff": "1.0 * nanometers",
            "ewaldErrorTolerance": 0.0005,
            "constraints": "HBonds",
            "rigidWater": True,
            "constraintTolerance": 0.00001,
            "solvate": True,
        },
        "integrator_params": {
            "integrator_type": "LangevinMiddle",
            "Temperature": "300 * kelvin",
            "Friction": "1.0 / picoseconds",
            "Timestep": "0.002 * picoseconds",
            "Pressure": "1.0 * bar",
        },
        "simulation_params": {
            "Ensemble": "NVT",
            "Number of Steps": 10000,
            "record_interval_steps": 100,
            "record_params": ["step", "potentialEnergy", "temperature", "density"],
        },
    }


@pytest.fixture
def openmm_sim(get_registry, setup_run_input):
    return OpenMMSimulation(
        input_params=setup_run_input,
        path_registry=get_registry("raw", False),
        save=False,
        sim_id="test",
        pdb_id="test",
    )


def test_unit_to_string(openmm_sim):
    # Test with a simple unit
    assert openmm_sim.unit_to_string(5 * unit.nanometer) == "5*nanometer"

    # Test with a compound unit
    assert (
        openmm_sim.unit_to_string(2 * unit.kilocalorie_per_mole) == "2*kilocalorie/mole"
    )

    # Test with a unitless quantity
    assert openmm_sim.unit_to_string(10 * unit.dimensionless) == "10*dimensionless"


class ScriptContent:
    def __init__(
        self,
        pdb_path,
        forcefield_files,
        nonbonded_method,
        constraints,
        rigid_water,
        constraint_tolerance,
        nonbonded_cutoff,
        ewald_error_tolerance,
        hydrogen_mass,
        time_step,
        temperature,
        friction,
        ensemble,
        pressure,
        record_interval_steps,
        solvate,
        integrator_type,
    ):
        self.pdb_path = pdb_path
        self.forcefield_files = forcefield_files
        self.nonbonded_method = nonbonded_method
        self.constraints = constraints
        self.rigid_water = rigid_water
        self.constraint_tolerance = constraint_tolerance
        self.nonbonded_cutoff = nonbonded_cutoff
        self.ewald_error_tolerance = ewald_error_tolerance
        self.hydrogen_mass = hydrogen_mass
        self.time_step = time_step
        self.temperature = temperature
        self.friction = friction
        self.ensemble = ensemble
        self.pressure = pressure
        self.record_interval_steps = record_interval_steps
        self.solvate = solvate
        self.integrator_type = integrator_type


@pytest.fixture
def script_content_vars_1():
    return ScriptContent(
        pdb_path="test.pdb",
        forcefield_files="'amber14-all.xml', 'amber14/tip3p.xml'",
        nonbonded_method=NoCutoff,
        constraints="None",
        rigid_water=False,
        constraint_tolerance=None,
        nonbonded_cutoff=1.0,
        ewald_error_tolerance=0.0005,
        hydrogen_mass=None,
        time_step=0.002,
        temperature=300,
        friction=1,
        ensemble="NVT",
        pressure=None,
        record_interval_steps=1000,
        solvate=False,
        integrator_type="LangevinMiddle",
    )


@pytest.fixture
def script_content_1(script_content_vars_1, openmm_sim):
    return openmm_sim._construct_script_content(
        pdb_path=script_content_vars_1.pdb_path,
        forcefield_files=script_content_vars_1.forcefield_files,
        nonbonded_method=script_content_vars_1.nonbonded_method,
        constraints=script_content_vars_1.constraints,
        rigid_water=script_content_vars_1.rigid_water,
        constraint_tolerance=script_content_vars_1.constraint_tolerance,
        nonbonded_cutoff=script_content_vars_1.nonbonded_cutoff,
        ewald_error_tolerance=script_content_vars_1.ewald_error_tolerance,
        hydrogen_mass=script_content_vars_1.hydrogen_mass,
        time_step=script_content_vars_1.time_step,
        temperature=script_content_vars_1.temperature,
        friction=script_content_vars_1.friction,
        ensemble=script_content_vars_1.ensemble,
        pressure=script_content_vars_1.pressure,
        record_interval_steps=script_content_vars_1.record_interval_steps,
        solvate=script_content_vars_1.solvate,
        integrator_type=script_content_vars_1.integrator_type,
    )


def test_construct_script_content_script1(script_content_1, script_content_vars_1):
    assert f"pdb = PDBFile('{script_content_vars_1.pdb_path}')" in script_content_1
    assert (
        f"forcefield = ForceField({script_content_vars_1.forcefield_files})"
        in script_content_1
    )
    assert (
        f"nonbondedMethod = {script_content_vars_1.nonbonded_method}"
        in script_content_1
    )
    assert f"constraints = {script_content_vars_1.constraints}" in script_content_1
    assert f"rigidWater = {script_content_vars_1.rigid_water}" in script_content_1
    assert f"dt = {script_content_vars_1.time_step}" in script_content_1
    assert f"temperature = {script_content_vars_1.temperature}" in script_content_1
    assert f"friction = {script_content_vars_1.friction}" in script_content_1
    assert (
        f"dataReporter = StateDataReporter('log.txt', "
        f"{script_content_vars_1.record_interval_steps}" in script_content_1
    )
    assert (
        "simulation = Simulation(modeller.topology, system, integrator, platform)"
        in script_content_1
    )
    assert "simulation.minimizeEnergy()" in script_content_1


@pytest.fixture
def script_content_vars_2():
    return ScriptContent(
        pdb_path="test.pdb",
        forcefield_files="'amber14-all.xml', 'amber14/tip3p.xml'",
        nonbonded_method=PME,
        constraints="None",
        rigid_water=False,
        constraint_tolerance=None,
        nonbonded_cutoff=1.0,
        ewald_error_tolerance=0.0005,
        hydrogen_mass=None,
        time_step=0.002,
        temperature=300,
        friction=1,
        ensemble="NVT",
        pressure=None,
        record_interval_steps=1000,
        solvate=True,
        integrator_type="LangevinMiddle",
    )


@pytest.fixture
def script_content_2(script_content_vars_2, openmm_sim):
    return openmm_sim._construct_script_content(
        pdb_path=script_content_vars_2.pdb_path,
        forcefield_files=script_content_vars_2.forcefield_files,
        nonbonded_method=script_content_vars_2.nonbonded_method,
        constraints=script_content_vars_2.constraints,
        rigid_water=script_content_vars_2.rigid_water,
        constraint_tolerance=script_content_vars_2.constraint_tolerance,
        nonbonded_cutoff=script_content_vars_2.nonbonded_cutoff,
        ewald_error_tolerance=script_content_vars_2.ewald_error_tolerance,
        hydrogen_mass=script_content_vars_2.hydrogen_mass,
        time_step=script_content_vars_2.time_step,
        temperature=script_content_vars_2.temperature,
        friction=script_content_vars_2.friction,
        ensemble=script_content_vars_2.ensemble,
        pressure=script_content_vars_2.pressure,
        record_interval_steps=script_content_vars_2.record_interval_steps,
        solvate=script_content_vars_2.solvate,
        integrator_type=script_content_vars_2.integrator_type,
    )


def est_construct_script_content_script2(script_content_2, script_content_vars_2):
    assert (
        f"ewaldErrorTolerance = {script_content_vars_2.ewald_error_tolerance}"
        in script_content_2
    )
    assert "modeller.addSolvent(forcefield" in script_content_2
    assert (
        """
            system = forcefield.createSystem(modeller.topology,
            nonbondedMethod=nonbondedMethod,
            nonbondedCutoff=nonbondedCutoff, ewaldErrorTolerance=ewaldErrorTolerance,
            constraints=constraints, rigidWater=rigidWater)
            """
        in script_content_2
    )
