clear, clc
db_dir = '/home/ryang/cuda-workspace/DeepPain/data/cropped_videos/';
sc_dir = '/home/ryang/cuda-workspace/DeepPain/data/Sequence_Labels/OPR/';
out_dir = '/home/ryang/cuda-workspace/DeepPain/data/balanced_data_seq/';
% out_dir_0 = '/home/ryang/cuda-workspace/DeepPain/data/balanced_data_seq/0/';
% out_dir_1= '/home/ryang/cuda-workspace/DeepPain/data/balanced_data_seq/1/';
% out_dir_2='/home/ryang/cuda-workspace/DeepPain/data/balanced_data_seq/2/';
% out_dir_3 = '/home/ryang/cuda-workspace/DeepPain/data/balanced_data_seq/3/';
% out_dir_4 = '/home/ryang/cuda-workspace/DeepPain/data/balanced_data_seq/4/';
% out_dir_5 = '/home/ryang/cuda-workspace/DeepPain/data/balanced_data_seq/5/';

users=dir(db_dir);
users={users(3:end).name};

usc=dir(sc_dir);                     %user score, opi
usc={usc(3:end).name};

disp(pwd)
for fol=1:length(users)
    mkdir(fullfile(out_dir,users{fol}))
%     mkdir(fullfile(out_dir_0,users{fol}))
%     mkdir(fullfile(out_dir_1,users{fol}))
%     mkdir(fullfile(out_dir_2,users{fol}))
%     mkdir(fullfile(out_dir_3,users{fol}))
%     mkdir(fullfile(out_dir_4,users{fol}))
%     mkdir(fullfile(out_dir_5,users{fol}))

    vids=dir(fullfile(db_dir,users{fol}));
    vids={vids(3:end).name};

    vidsc=dir(fullfile(sc_dir,usc{fol}));
    vidsc={vidsc(3:end).name};
    
    num_np = 0;
    num_p = 0;
    idxp = [];
    idxnp = [];
    for v=1:length(vids)

        load(fullfile(db_dir,users{fol},vids{v}));           %load faceVid
        fprintf('%d ',v)
        %[x,y,nFrms]=size(faceVid);
        
        vs=dir(fullfile(sc_dir,usc{fol},'*.txt'));
        OPI = load(fullfile(sc_dir,usc{fol},vs(v).name));   %load score 

        if OPI == 0
           num_np = num_np+1
           idxnp = [idxnp;v];
           %save(fullfile(out_dir_0,users{fol},vids{v}),'faceVid')
        end
        
        if OPI > 0
           num_p = num_p + 1;
           idxp = [idxp;v];
           %save(fullfile(out_dir_1,users{fol},vids{v}),'faceVid')
        end      
    end   

    if num_np > num_p
       for i =1:num_p
           save(fullfile(out_dir,users{fol},vids{idxp(i)}),'faceVid')
           save(fullfile(out_dir,users{fol},vids{idxnp(i)}),'faceVid')
       end
       %save(fullfile(out_dir,users{fol},vids{idxnp(1:num_p)}),'faceVid')
    else
           save(fullfile(out_dir,users{fol},vids{idxp(1:num_p)}),'faceVid')
           save(fullfile(out_dir,users{fol},vids{idxnp(1:num_np)}),'faceVid')
    end
    
end 
