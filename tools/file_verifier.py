 files = sorted(listdir(os.getcwd()))


# Verify files in directory are of the correct format.
 for f in range(len(files)):
     if (f-2) % 7 == 0 and not files[f].endswith('target.nrrd'):
         print(files[f])
         break
