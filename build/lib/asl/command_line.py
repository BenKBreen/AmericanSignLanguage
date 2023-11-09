
# imports
import sys
import asl
import subprocess

# main script
def aslread():

    # path
    path = sys.argv[1]
    
    print('=' * 20)
    print('Annotating Video')
   
    # create the video 
    V = asl.video_file(path)
    
    print('=' * 20)
    print('Writing video to file')

    # write to file
    file_path = asl.write_video(V)

    # show the video object
    subprocess.call(['open', file_path])
