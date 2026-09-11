function make_combined_from_workspace(ws_path, ref_folder, out_folder, base_config_dir)
%MAKE_COMBINED_FROM_WORKSPACE  Build CombinedData.mat from a saved full Verasonics workspace.
%
%   make_combined_from_workspace(WS_PATH, REF_FOLDER, OUT_FOLDER, BASE_CONFIG_DIR)
%
%   Recovery path for an acquisition whose runtime AcquisitionParametersAndECG.mat was not
%   written, but for which a full session workspace dump exists (saved with the sequence still
%   loaded, so SW/TPC/TX/Receive already describe that acquisition). The workspace is the merged
%   constant+dynamic state, i.e. exactly what CombinedData.mat is -- it only lacks the RF binary
%   layout (RF_rows/RF_cols/RF_frames) and NonzeroRFcolumns, which are copied from a sibling
%   acquisition of the same sequence (REF_FOLDER) and the base config directory.
%
%   Used for the 2026-08-18 in-vivo 61-element / 1900-cycle / 25 V measurement.

if nargin < 4 || isempty(base_config_dir)
    base_config_dir = 'D:\Luuk van Knippenberg\SWI\Base config files';
end

out_mat = fullfile(out_folder, 'CombinedData.mat');
fprintf('copying workspace -> %s\n', out_mat);
copyfile(ws_path, out_mat);

S = load(fullfile(ref_folder, 'AcquisitionParametersAndECG.mat'), ...
         'RF_rows', 'RF_cols', 'RF_frames');
RF_rows   = S.RF_rows;
RF_cols   = S.RF_cols;
RF_frames = S.RF_frames;
load(fullfile(base_config_dir, 'NonzeroRFcolumns.mat'), 'NonzeroRFcolumns');

clear S ws_path ref_folder out_folder base_config_dir
save(out_mat, 'RF_rows', 'RF_cols', 'RF_frames', 'NonzeroRFcolumns', '-append');
fprintf('done: RF_frames = %s\n', mat2str(RF_frames));
end
