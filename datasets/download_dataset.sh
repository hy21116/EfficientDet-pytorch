start=`date +%s`

# Download VOC2007 DATASET
echo "Downloading VOC2007 dataset ... "
wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar 
wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar

# Extract tar files
echo "Extracting trainval ... "
tar -xf train/VOCtrainval_06-Nov-2007.tar 
echo "Extracting test ... "
tar -xf test/VOCtest_06-Nov-2007.tar 

# Remove tar files
echo "removing tars ... "
rm -rf test/VOCtest_06-Nov-2007.tar
rm -rf train/VOCtrainval_06-Nov-2007.tar

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
ㅎ