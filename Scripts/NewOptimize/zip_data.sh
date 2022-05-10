
echo 'Zipping data...'

zip -ru (date +'results-%y-%m-%d.zip') results/*.pkl vpt_results/*.pkl

echo 'Done'
