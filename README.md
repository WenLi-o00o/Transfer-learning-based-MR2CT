1. All training MR image data should be put in "/datasets/DcmData/trainA/" folder;
    Your training CT image data should be put in "/datasets/DcmData/trainB/" folder;
    Testing MR image data should be put in "/datasets/DcmData/testA/" folder;
    Testing CT image data should be put in "/datasets/DcmData/testB/" folder.

2. The mr_max, mr_min, ct_max, ct_min of your dataset should be specified in MR2CT.sh before runing the code;
    In ManyTo1.py, you need to specify the ct_max and ct_min again, the value should be same as step 2.1. you also need to modify the path to your computer path.

3. In config.json, you need to change the path to your computer path.

4. This code can only test one patient each time at present, please remember to update corresponding testA and testB data each time for new patient, and export the generated results from "../results/generated_dicom/".

 