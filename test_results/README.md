## to plot 2cycle
python3 plot_data.py --outfile 2cycle.png --metric "sum_2cycles" --normalize --title "Number of 2 Cycles compared with the Default Algo (as a percentage)"

## to plot quality
python3 plot_data.py --outfile quality.png --metric "Quality" --title "Quality of Matching compared with the Default Algo (as a percentage)"

## to plot runtime

python3 plot_data.py --outfile runtime.png --metric "running time" --title "Runtime compared with the Default Algo (as a percentage)" --normalize

## to plot coauthorship stuff

python3 plot_data.py --outfile coauthor.png --metric "sum_coauthors" --title "Number of coauthors compared with the Default Algo (as a percentage)" --normalize --title-size 24

## DCX: I add some algorithms. now the figure format becomes crashed.. maybe need to ask Claude again.. sorry!!

## to plot region
# DCX: the current metric seems stupid.. see the modification in collect_records function. could MC help figure out a better way?

python3 plot_data.py --outfile region.png --metric "Region" --title "Average Different Reviewer Regions for a Paper"

## to plot seniority

python3 plot_data.py --outfile seniority.png --metric "Seniority" --title "Number of papers without senior reviewers compared with the Default Algo (as a percentage)" --normalize

## to plot entropy
#DCX: for this thing Ours and Ours_with_Original_Sampling is the same. TODO: need to delete Ours_with_Original_Sampling