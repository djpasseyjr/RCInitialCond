
echo 'Collecting plots...'

if [ -f `date +"figures-%y-%m-%d.zip"` ]
then
	rm `date +"figures-%y-%m-%d.zip"`
fi


zip -rq `date +"figures-%y-%m-%d.zip"` plots.html attractor/ histograms/

echo 'Done'
