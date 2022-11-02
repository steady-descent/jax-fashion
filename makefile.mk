conda-env:
	conda create -n jax-fashion python=3.9
	conda activate jax-fashion
	pip install -r requirements.txt

run:
	python main_mlp.py