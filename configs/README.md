# Turing Job Commands

## Submit Jobs

```bash
# short config (quick test => 5 tracks)
sbatch configs/turing.sh --train-config configs/vocals_short.yaml --stem vocals --test-type quick

# medium config (full test split)
sbatch configs/turing.sh --train-config configs/vocals_medium.yaml --stem vocals --test-type full

# long config (full test split)
sbatch configs/turing.sh --train-config configs/vocals_long.yaml --stem vocals --test-type full

# override number of test tracks
sbatch configs/turing.sh --train-config configs/vocals_medium.yaml --stem vocals --test-type full --max-test-tracks 10
```

## Check Queue / Job Status

```bash
# jobs currently queued/running for your user
squeue -u "$USER"

# detailed status after completion/failure
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList,Reason
```

## Monitor Logs (out/err)

```bash
# live output log
watch -n 2 "tail -n 80 logs/moises_light_vocals_<JOB_ID>.out"

# live error log
watch -n 2 "tail -n 80 logs/moises_light_vocals_<JOB_ID>.err"

# if you only need one stream, use plain tail -f
tail -f logs/moises_light_vocals_<JOB_ID>.out
```

## Monitor GPU Usage

```bash
# run while job is active on allocated node
watch -n 2 nvidia-smi
```

