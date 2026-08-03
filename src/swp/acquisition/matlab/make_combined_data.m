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
    % Phantom: shorter 2-frame base config (no cardiac dependency).
    copyfile(fullfile(base_config_dir, 'S5_1_SWI_PulseInversion_P15-xx_runtime_2frames.mat'), ...
             'CombinedData.mat');
end

% TX (constant) from the base file; dynamic parameters overwrite the rest.
load('CombinedData.mat', 'TX')
load('AcquisitionParametersAndECG.mat')

if ~invivo_data
    SW.Nframes = 2;
    Resource.RcvBuffer(2).numFrames = 2;
    Resource.RcvBuffer(2).lastFrame = 2;
    Resource.RcvBuffer(5).numFrames = 2;
    Resource.RcvBuffer(5).lastFrame = 2;
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
