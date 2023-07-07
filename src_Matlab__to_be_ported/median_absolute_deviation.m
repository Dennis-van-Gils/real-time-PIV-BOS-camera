function median_absolute_deviation()
clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
imtool close all;  % Close all imtool figures.
clear;  % Erase all existing variables.
workspace;  % Make sure the workspace panel is showing.
format longg;
format compact;
fontSize = 20;

% Change the current folder to the folder of this m-file.
if(~isdeployed)
	cd(fileparts(which(mfilename)));
end
	
% Check that user has the Image Processing Toolbox installed.
hasIPT = license('test', 'image_toolbox');
if ~hasIPT
	% User does not have the toolbox installed.
	message = sprintf('Sorry, but you do not seem to have the Image Processing Toolbox.\nDo you want to try to continue anyway?');
	reply = questdlg(message, 'Toolbox missing', 'Yes', 'No', 'Yes');
	if strcmpi(reply, 'No')
		% User said No, so exit.
		return;
	end
end

% Read in a standard MATLAB gray scale demo image.
folder = fullfile(matlabroot, '\toolbox\images\imdemos');
baseFileName = 'coins.png';
% Get the full filename, with path prepended.
fullFileName = fullfile(folder, baseFileName);
% Check if file exists.
if ~exist(fullFileName, 'file')
	% File doesn't exist -- didn't find it there.  Check the search path for it.
	fullFileName = baseFileName; % No path this time.
	if ~exist(fullFileName, 'file')
		% Still didn't find it.  Alert user.
		errorMessage = sprintf('Error: %s does not exist in the search path folders.', fullFileName);
		uiwait(warndlg(errorMessage));
		return;
	end
end
grayImage = imread(fullFileName);
% Get the dimensions of the image.  
% numberOfColorBands should be = 1.
[rows, columns, numberOfColorBands] = size(grayImage)
if numberOfColorBands > 3
	% It's not really gray scale, it's color, so convert it to gray scale.
	grayImage = rgb2gray(grayImage);
end
% Display the original gray scale image.
subplot(2, 2, 1);
imshow(grayImage, []);
title('Original Grayscale Image', 'FontSize', fontSize);
% Enlarge figure to full screen.
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
% Give a name to the title bar.
set(gcf,'name','MAD noise removal Demo','numbertitle','off') 

% Let's compute and display the histogram.
[pixelCount, grayLevels] = imhist(grayImage);
subplot(2, 2, 2); 
bar(grayLevels, pixelCount);
grid on;
title('Histogram of Original Image', 'FontSize', fontSize);
xlim([0 grayLevels(end)]); % Scale x axis manually.

% Add Salt and Pepper noise to create outliers.
noisyImage = imnoise(grayImage, 'salt & Pepper', .01);
noisyImage(1,3) = 255;
% Display the image.
subplot(2, 2, 3);
imshow(noisyImage);
title('Noisy Image', 'FontSize', fontSize);

% Extract the noise that was added.
% This is where the noisy image values don't exactly match the original noise-free values.
theNoise = noisyImage ~= grayImage;

% Display the original noise that we added.
subplot(2, 2, 4);
imshow(theNoise, []);
numberOfNoisePixels = sum(theNoise(:));
caption = sprintf('The Actual Noise - %d Outlier Pixels\n(both black and white show as white)', numberOfNoisePixels);
title(caption, 'FontSize', fontSize);


% promptMessage = sprintf('Now we will compute the MAD image.\nDo you want to Continue or Quit?');
% titleBarCaption = 'Continue?';
% button = questdlg(promptMessage, titleBarCaption, 'Continue', 'Quit', 'Continue');
% if strcmp(button, 'Quit')
% 	return;
% end

% Ask user for a number.
global madRatioThreshold;
defaultValue = 5;
titleBar = 'Enter a ratio';
userPrompt = sprintf('Enter the ratio of the MAD value to the image value.\nA lower value assigns more pixels as outliers and does more noise removal.\nA higher value assigns fewer pixels as outliers and does less noise removal.');
caUserInput = inputdlg(userPrompt, titleBar, 1, {num2str(defaultValue)});
if isempty(caUserInput),return,end; % Bail out if they clicked Cancel.
% Round to nearest integer in case they entered a floating point number.
madRatioThreshold = round(str2double(cell2mat(caUserInput)));
% Check for a valid integer.
if isnan(madRatioThreshold)
    % They didn't enter a number.  
    % They clicked Cancel, or entered a character, symbols, or something else not allowed.
    madRatioThreshold = defaultValue;
    message = sprintf('I said it had to be an integer.\nI will use %d and continue.', madRatioThreshold);
    uiwait(warndlg(message));
end

%==========  MAD Filter  ==================================================
% Block process the image to do the MAD Filter.
windowSize = 7;
% Each 3x3 block will get replaced by one value.
% Output image will be smaller by a factor of windowSize.
myFilterHandle = @MAD_Filter;
MADValueImage = nlfilter(noisyImage,[windowSize, windowSize], myFilterHandle);
%============================================================


% Display the "truth" noise image again.
figure;
% Display the original noise that we added.
subplot(2, 2, 1);
imshow(theNoise, []);
numberOfNoisePixels = sum(theNoise(:));
caption = sprintf('The Actual Noise\n%d Noise Pixels', numberOfNoisePixels);
title(caption, 'FontSize', fontSize);

% Display the outliers as detected by MAD filter.
subplot(2, 2, 2);
imshow(MADValueImage, []);
title('MAD Filtered Image (Outliers)', 'FontSize', fontSize);
% Enlarge figure to full screen.
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
% Give a name to the title bar.
set(gcf,'name','MAD noise removal Demo','numbertitle','off') 

% Display the original, noisy image.
subplot(2, 2, 3);
imshow(noisyImage);
title('Original, Noisy Image', 'FontSize', fontSize);

% Repair the image by replacing the outliers pixels with "good" pixels.
% Get the median filter image.  This will be the "good image" to harvest pixel values from.
medianImage = medfilt2(noisyImage, [3 3]);
% Let's replace the noisy image pixels at the outlier locations
% with the pixels from the median filtered image at the outlier locations.
repairedImage = noisyImage; % Initialize.
repairedImage(MADValueImage) = medianImage(MADValueImage); % Replacement.
% Display the repaired image.
subplot(2, 2, 4);
imshow(repairedImage, []);
title('Final, Repaired Output Image', 'FontSize', fontSize);


%=========================================================
% Takes one n by m 2-D block of image data and gets the
% Median Absolute Deviation
function output = MAD_Filter(pixelsInBlock)
	global madRatioThreshold;
	try
		% Assign default value.
		output = 0;
		
		% Get the median of those values.
		medianValue = median(pixelsInBlock(:));
		
		% Get the absolute deviation
		absoluteDeviation = abs(single(pixelsInBlock) - single(medianValue));

		% Get the median of those values.
		% This is the "Median Absolute Deviation" value.
		MAD_Value = uint8(median(absoluteDeviation(:)));
		
		% Determine if it's an outlier
		middleIndex = ceil(numel(pixelsInBlock) / 2);
		centralValue = absoluteDeviation(middleIndex);
		% If the central value of the absolute deviations is more than
		% some factor times the MAD_Value, then it's an outlier.
		if centralValue > madRatioThreshold * MAD_Value % && centralValue > 0
			itIsAnOutlier = true;
		else
			itIsAnOutlier = false;			
		end

		% Assign this to our output argument.
% 		output = MAD_Value;
		output = itIsAnOutlier;
% 		output = [MAD_Value 255 * uint8(itIsAnOutlier)];
% 		output = uint16(MAD_Value) + uint16(bitshift(itIsAnOutlier, 8));

	catch ME
		% Some kind of problem...
		errorMessage = sprintf('Error in MAD_Filter():\n\nError Message:\n%s', ME.message);
		% uiwait(warndlg(errorMessage));
		fprintf(1, '%s\n', errorMessage);
	end
	return; 


