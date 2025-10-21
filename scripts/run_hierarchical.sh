python scripts/run_hierarchical_benchmark.py --task hierarchical_two_moons --algorithm snpe --num_simulations 1000 --num_observation 1 --output_dir test_results --seed 42 --num_samples 100
python scripts/run_hierarchical_benchmark.py --task hierarchical_two_moons --algorithm snpe --num_simulations 5000 --num_observation 1 --output_dir test_results --seed 42 --num_samples 100
python scripts/run_hierarchical_benchmark.py --task hierarchical_two_moons --algorithm snpe --num_simulations 10000 --num_observation 1 --output_dir test_results --seed 42 --num_samples 100
python scripts/visualalize_hierarchical_posterior.py --task hierarchical_two_moons 
