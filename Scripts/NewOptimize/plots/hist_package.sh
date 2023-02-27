
echo 'Collecting plots...'

if [ -f `date +"histograms-%y-%m-%d.zip"` ]
then
	rm `date +"histograms-%y-%m-%d.zip"`
fi


zip -rq `date +"histograms-%y-%m-%d.zip"` hist-plots.html histograms/

echo 'Done'
