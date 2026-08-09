function make_combined_data(data_folder, base_config_dir)
%MAKE_COMBINED_DATA  Build CombinedData.mat from the base config + dynamic params.
%
%   make_combined_data(DATA_FOLDER, BASE_CONFIG_DIR)
%
%   The Verasonics acquisition only saves the *dynamic* parameters at runtime
%   (AcquisitionParametersAndECG.mat); the *constant* parameters live in a base
%   config runtime .mat. This merges the two into CombinedData.mat -- the v7.3
%   workspace the beamformer reads -- exactly as the interactive post-processing
%   script does. Ported for the shearWaveProcessing pipeline (called from Python
%   via `matlab -batch`); the interactive uigetdir is replaced by DATA_FOLDER.
%
%   Phantom vs in-vivo is detected from RF_frames(1) (==2 -> phantom, 2 frames).

if nargin < 2 || isempty(base_config_dir)
    base_config_dir = 'D:\Luuk van Knippenberg\SWI\Base config files';
end

cd(data_folder);

load('AcquisitionParametersAndECG.mat', 'RF_frames')
if RF_frames(1) == 2
    invivo_data = false;
else
    invivo_data = true;
end

% Copy the matching base config (constant parameters) to CombinedData.mat.
if invivo_data
    copyfile(fullfile(base_config_dir, 'S5_1_SWI_PulseInversion_P15-xx_runtime.mat'), ...
             'CombinedData.mat');
else
    % Phantom parameter sweep: pick the base config matching this run's push settings, since the
    % Receive/RcvBuffer frame structure differs per config (10 pushes; Ndetect varies with PRI).
    % Naming: BaseConfig_10frames_<cycles>cycles_<elements>elements_<pri>PRI.mat (in PhantomSweep/).
    load('AcquisitionParametersAndECG.mat', 'SW');
    bname = sprintf('BaseConfig_10frames_%dcycles_%delements_%dPRI.mat', ...
                    round(SW.pushCycle), round(SW.nb_push_elmts), round(SW.PRI_us));
    bpath = fullfile(base_config_dir, 'PhantomSweep', bname);
    assert(exist(bpath, 'file') == 2, 'PhantomSweep base config not found: %s', bpath);
    copyfile(bpath, 'CombinedData.mat');
end

% TX (constant) from the base file; dynamic parameters overwrite the rest.
load('CombinedData.mat', 'TX')
load('AcquisitionParametersAndECG.mat')

if ~invivo_data
    % Use the ACTUAL number of pushes from the runtime params. Older phantom acquisitions had 2, but
    % the parameter-sweep data has 10 (and Ndetect varies with PRI - both already correct in the loaded
    % runtime SW). Do NOT hardcode to 2 (that discarded pushes 3-10).
    nfr = double(SW.Nframes);
    Resource.RcvBuffer(2).numFrames = nfr;
    Resource.RcvBuffer(2).lastFrame = nfr;
    Resource.RcvBuffer(5).numFrames = nfr;
    Resource.RcvBuffer(5).lastFrame = nfr;
end

% Merge the dynamic push transmits (TX_tmp) into the tail of the base TX array.
TX(end-length(TX_tmp)+1:end) = TX_tmp;
clear TX_tmp;

% Per-buffer saved-channel masks for reconstructing the external RF binaries.
load(fullfile(base_config_dir, 'NonzeroRFcolumns.mat'), 'NonzeroRFcolumns')

% Append the (now dynamic-overwritten) workspace to the copied base config.
% TX_tmp is dropped (as in the original post-processing script) along with the
% function's own path argument; every dynamic variable -- including
% RF_frames/RF_rows/RF_cols, which the reader needs to reconstruct the external
% RF binaries -- is written into CombinedData.mat.
clear base_config_dir
save('CombinedData.mat', '-append');
end
