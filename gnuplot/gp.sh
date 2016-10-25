#gnuplot -e "input_file='/home/aerolabio/librealsense/examples/librealsense_feat/gnuplot/pos-time.csv'" /home/aerolabio/librealsense/examples/librealsense_feat/gnuplot/gp_timeStats_popup.pg
gnuplot -e "input_file='$1'" gp_timeStats_popup.pg
