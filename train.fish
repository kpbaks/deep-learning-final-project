#!/usr/bin/env fish

function hr
    string repeat --count $COLUMNS -
end

if not test -f train.py
    echo "train.py not found"
    exit 1
end

set -l generator_learning_rates 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001 0.00005 0.00002 0.00001 0.000005 0.000002 0.000001
set -l discriminator_learning_rates $generator_learning_rates

# set -l num_combinations (math "$(count $generator_learning_rates) * $(count $discriminator_learning_rates)")
# set -l num_epochs (math "floor(3000 / $num_combinations)")
set -l num_epochs 30
set -l save_model_every 5
set -l batch_size 32

set --local # Print local variables

for glr in $generator_learning_rates
    for dlr in $discriminator_learning_rates
        test $dlr -gt $glr; and continue
        hr
        set -l command "./train.py --glr $glr --dlr $dlr --epochs $num_epochs --save-model-every $save_model_every --batch-size $batch_size"
        echo $command | fish_indent --ansi
        eval $command
        test $status -eq 0; and begin
            echo "Failed"
        end
    end
end
