
echo 'Zipping data...'

zip -r `date +"results-%y-%m-%d.zip"` results/*.pkl vpt_results/*.pkl attractor_results/*.pkl traintimes/vpts/*.pkl traintimes/optimization/*.pkl

echo 'Done'
