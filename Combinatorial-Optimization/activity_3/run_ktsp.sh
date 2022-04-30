underline="_"
arq_saida="output_instance_v"
filename="mo824_atividade2_coords.txt"
declare -i v2=0
for ((v=100; v<=250; v+=50))
do
v2=$(( $v / 2 ))
python3 ktsp_lagragian_relaxation.py $v 0 $filename > output_instances/V$underline$v/$arq_saida$v$underline"k0".txt
python3 ktsp_lagragian_relaxation.py $v $v $filename > output_instances/V$underline$v/$arq_saida$v$underline"k"$v.txt
done
for ((v=100; v<=250; v+=50))
do
v2=$(( $v / 2 ))
python3 ktsp_lagragian_relaxation.py $v $v2 $filename > output_instances/V$underline$v/$arq_saida$v$underline"k"$v2.txt
done


# python3 ktsp_lagragian_relaxation.py 100 0 mo824_atividade2_coords.txt