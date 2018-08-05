

OCT = exist('OCTAVE_VERSION') ~= 0;           % check if we run Matlab or Octave

me = mfilename;                                            % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me));        % where am I located
if OCT && numel(mydir)==2 
  if strcmp(mydir,'./'), mydir = [pwd,mydir(2:end)]; end
end                 % OCTAVE 3.0.x relative, MATLAB and newer have absolute path

addpath(mydir(1:end-1))

addpath([mydir,'diag'])
addpath([mydir,'dist'])
addpath([mydir,'gp'])
addpath([mydir,'misc'])
addpath([mydir,'optim'])
addpath([mydir,'NN'])
addpath([mydir,'util'])
addpath([mydir,'objective'])
addpath([mydir,'acquisition'])





clear me mydir