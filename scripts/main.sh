Base_model=$1
RL_dataset=$2

echo "Using $RL_dataset for self-training"

bash scripts/questioner_train.sh $Base_model $Base_model ${RL_dataset}_questioner_v1
bash scripts/solver_train.sh $Base_model /root/autodl-tmp/models/${RL_dataset}_questioner_v1 ${RL_dataset}_solver_v1

for i in {2..3}; do
    prev=$((i-1))

    bash scripts/questioner_train.sh \
        /root/autodl-tmp/models/${RL_dataset}_solver_v${prev} \
        /root/autodl-tmp/models/${RL_dataset}_questioner_v${prev} \
        ${RL_dataset}_questioner_v${i}

    bash scripts/solver_train.sh \
        /root/autodl-tmp/models/${RL_dataset}_solver_v${prev} \
        /root/autodl-tmp/models/${RL_dataset}_questioner_v${i} \
        ${RL_dataset}_solver_v${i}
done

