"""ProcTHOR house -> scene graph -> problem.pddl -> Fast Downward plan.

    uv run python -m scenegraph_pddl.scripts.run_pipeline --seed 3 --goal apple fridge
    uv run python -m scenegraph_pddl.scripts.run_pipeline --seed 7036 --num-random-goals 3 --planner ff-wastar2

Writes scene_graph.json, scene_graph.png, domain.pddl, problem.pddl, plan.txt
and fast_downward.log to <output-dir>/seed_<seed>/.
"""
import argparse
import os
import shutil

from scenegraph_pddl import config
from scenegraph_pddl.house import ProcTHORHouse
from scenegraph_pddl.pddl import fast_downward, problem
from scenegraph_pddl.scene_graph import add_node_features, build_scene_graph, save_scene_graph


def load_scene(seed):
    """Load house `seed` and build its scene graph (with node features)."""
    house = ProcTHORHouse(seed)
    # The simulator is only needed to build the occupancy grid
    house.close()
    graph = add_node_features(build_scene_graph(house))
    return house, graph


def run_pipeline(seed, goals=None, num_random_goals=1,
                 output_dir=config.PIPELINE_OUTPUT_DIR, planner=fast_downward.DEFAULT_PLANNER):
    """Run all three steps for one house. Returns (graph, goals, actions, cost)."""
    output_dir = os.path.join(output_dir, f'seed_{seed}')
    os.makedirs(output_dir, exist_ok=True)

    # 1. House -> scene graph
    house, graph = load_scene(seed)
    save_scene_graph(graph, output_dir)
    print(f"Scene graph: {len(graph['nodes'])} nodes, "
          f"{len(graph['containers'])} containers, {len(graph['objects'])} objects")

    # 2. Scene graph -> problem.pddl
    goals = problem.resolve_goals(graph, goals or problem.random_goals(
        graph, num_goals=num_random_goals, seed=seed))
    print(f'Goal: {goals}')
    domain_path = os.path.join(output_dir, 'domain.pddl')
    problem_path = os.path.join(output_dir, 'problem.pddl')
    shutil.copy(problem.DOMAIN_PATH, domain_path)
    motion_costs = problem.compute_motion_costs(house, graph)
    with open(problem_path, 'w') as f:
        f.write(problem.build_problem(graph, goals, motion_costs))

    # 3. Solve with Fast Downward
    actions, cost = fast_downward.solve(
        domain_path, problem_path, planner=planner,
        plan_path=os.path.join(output_dir, 'plan.txt'),
        log_path=os.path.join(output_dir, 'fast_downward.log'))
    if actions is not None:
        print(f'Plan ({len(actions)} steps, cost {cost}):')
        for action in actions:
            print('  ' + ' '.join(action))
    return graph, goals, actions, cost


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=0, help='ProcTHOR-10k house index (0-9999)')
    parser.add_argument('--goal', nargs=2, action='append', metavar=('OBJECT', 'CONTAINER'),
                        help="put OBJECT in CONTAINER; names are PDDL names ('apple_23') "
                             "or generic names ('apple'). Repeatable. Random if omitted.")
    parser.add_argument('--num-random-goals', type=int, default=1,
                        help='number of random goals when no --goal is given')
    parser.add_argument('--output-dir', default=config.PIPELINE_OUTPUT_DIR)
    parser.add_argument('--planner', default=fast_downward.DEFAULT_PLANNER,
                        help=f'one of {list(fast_downward.PLANNERS)} (ff-wastar2 is much faster '
                             'on larger houses) or a Fast Downward search string')
    return parser.parse_args()


def main():
    args = parse_args()
    _, _, actions, _ = run_pipeline(args.seed, args.goal, args.num_random_goals,
                                    args.output_dir, args.planner)
    if actions is None:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
