"""Install and run the Fast Downward planner (https://www.fast-downward.org).

Fast Downward is cloned into third_party/fast-downward (override with the
FAST_DOWNWARD_DIR environment variable) and built on first use. Building needs
git, CMake and a C++20 compiler.
"""
import os
import re
import subprocess
import sys

from procgraph import config

REPOSITORY = 'https://github.com/aibasel/downward.git'
VERSION = 'release-24.06.1'

# Named search configurations, following pddlstream's planner names. Any other
# planner string is passed to Fast Downward as --search.
PLANNERS = {
    'ff-astar': ['--evaluator', 'h=ff()', '--search', 'astar(h)'],
    'ff-wastar2': ['--evaluator', 'h=ff(transform=adapt_costs(cost_type=plusone))',
                   '--search', 'lazy_wastar([h],preferred=[h],reopen_closed=true,boost=100,w=2,'
                               'preferred_successors_first=true,cost_type=plusone)'],
}
DEFAULT_PLANNER = 'ff-astar'


def ensure_installed(install_dir=config.FAST_DOWNWARD_DIR):
    """Clone and build Fast Downward unless already done. Returns the path to fast-downward.py."""
    driver = os.path.join(install_dir, 'fast-downward.py')
    if not os.path.exists(driver):
        print(f'Cloning Fast Downward {VERSION} into {install_dir}')
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', VERSION,
                        REPOSITORY, install_dir], check=True)

    if not os.path.exists(os.path.join(install_dir, 'builds', 'release', 'bin', 'downward')):
        print('Building Fast Downward (one time, takes a few minutes)')
        subprocess.run([sys.executable, 'build.py'], cwd=install_dir, check=True)

    return driver


def parse_plan(plan_path):
    """Read a Fast Downward plan file. Returns (actions, cost); each action is a
    tuple such as ('pick', 'robot1', 'apple_23', 'fridge_5', 'hand1')."""
    actions = []
    cost = None
    with open(plan_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(';'):
                match = re.search(r'cost = (\d+(\.\d+)?)', line)
                if match:
                    cost = float(match.group(1))
            elif line:
                actions.append(tuple(line.strip('()').split()))
    return actions, cost


def solve(domain_path, problem_path, plan_path='sas_plan', planner=DEFAULT_PLANNER,
          timeout=None, log_path=None, install_dir=config.FAST_DOWNWARD_DIR):
    """Solve a PDDL task. planner is a name from PLANNERS (use 'ff-wastar2' for
    larger houses) or a Fast Downward search string such as 'astar(lmcut())'.

    Returns (actions, cost) as parse_plan does, or (None, None) if no plan was found.
    """
    driver = ensure_installed(install_dir)
    if os.path.exists(plan_path):
        os.remove(plan_path)

    command = [sys.executable, os.path.abspath(driver),
               '--plan-file', os.path.abspath(plan_path),
               os.path.abspath(domain_path), os.path.abspath(problem_path),
               *PLANNERS.get(planner, ['--search', planner])]
    # Run next to the plan so the translator's output.sas lands there too
    result = subprocess.run(command, cwd=os.path.dirname(os.path.abspath(plan_path)),
                            capture_output=True, text=True, timeout=timeout)

    if log_path is not None:
        with open(log_path, 'w') as f:
            f.write(result.stdout + result.stderr)

    if not os.path.exists(plan_path):
        print(result.stdout[-3000:] + result.stderr[-3000:])
        print(f'Fast Downward found no plan (exit code {result.returncode})')
        return None, None
    return parse_plan(plan_path)
