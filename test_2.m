clear all;
close all;
PATH = '/Users/nxs162330/Documents/Clean_Speech/Male';
noiseFiles  = dir(PATH);
noiseFiles = noiseFiles(~ismember({noiseFiles.name},{'.' '..' '.DS_Store'}));
nNoiseFiles  = numel(noiseFiles);
fs = 8000;
framesize = 40;
nFFT = 512;
Nofilters = 26;
for num = 13%1:nNoiseFiles
    noiseFile           = [noiseFiles(num).folder '/' noiseFiles(num).name];
    [x,fs]   = audioread(noiseFile);
    x_test = zeros(fs*3,1);
    k=1;
    framelen=floor(framesize * fs/1000); % Frame size in samples
    if rem(framelen,2)==1, framelen=framelen+1; end;
    PERC=50; % window overlap in percent of frame size
    overlapplen=floor(framelen*PERC/100);
    framelen2=framelen-overlapplen; % update rate in samples
    Nframes=floor(24000/framelen2)-1;
    if(length(x)<=24001)
        x_test(1:length(x))=x;
        mfcc = mel_feature(x_test,fs,framesize,nFFT,Nofilters);
        f0 = pitch(x_test,fs,'WindowLength',round(fs*0.04),'OverlapLength',round(fs*(0.04-0.02)));
        pitch_final = mean(f0);
        mfcc_final = mean(mfcc);
%         [formants] = formantsmatlabu(x_test,fs,framesize);
%         formants_final = mean(formants);
    else
        Nframe = floor(length(x)/(fs*3));
        for i =1:Nframe
            x_test = x(k:k+(fs*3)-1);
            mfcc = mel_feature(x_test,fs,framesize,nFFT,Nofilters);
            f0 = pitch(x_test,fs,'WindowLength',round(fs*0.04),'OverlapLength',round(fs*(0.04-0.02)));
%             [formants] = formantsmatlabu(x_test,fs,framesize);
            pitch_final(i,:) = mean(f0);
            mfcc_final(i,:) = mean(mfcc);
%             formants_final(i,:) = mean(formants);
            k=k+(fs*3);
        end
    end
   % final = cat(3,mfcc_final,pitch_final);

    final = [mfcc_final pitch_final];
    if(num==13)
        output = final;
    end
%     if(num>1)
%         output = cat(1,output,final);
%     end
end
filename_write = strcat('/Users/nxs162330/Documents/Clean_Speech/M_test_', num2str(num),'.mat');
save(filename_write,'output','-v7.3');
