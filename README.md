
TO install:
```
pip install .
```
or for development:
```
pip install -e .
```

to run a simple 1v1 game:
```
python src\tron_app.py --mode human --players 2
```

Running continuous GA training:
``` Bash
python continuous_trainer.py --mode ga \
    --width 64 --height 48 \
    --hidden 24 --pop_size 64 --elite_count 8 \
    --eval_envs 1024 --checkpoint_steps 10 \
    --output_dir training_runs
```

Running continuous RL training:
``` Bash
python continuous_trainer.py --mode rl \
    --width 64 --height 48 --players 2 \
    --opponent greedy --checkpoint_steps 50000 \
    --learning_rate 3e-4 --n_steps 2048 \
    --output_dir training_runs
```