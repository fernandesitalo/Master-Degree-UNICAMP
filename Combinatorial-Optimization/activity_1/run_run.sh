underline="_"
arq_entrada="instance_"
arq_saida="output_instance"
for ((j=1; j<11; j++))
do
for ((i=100; i<=1000; i+=100))
do
python3 main_problem.py $arq_entrada$i$underline$j > output_instances/J$underline$i/$arq_saida$i$underline$j.txt
done
done
