function output = MAD_Filter(pixelsInBlock)
  %=========================================================
  % Takes one n by m 2-D block of image data and gets the
  % Median Absolute Deviation
  %
  % Source: http://nl.mathworks.com/matlabcentral/answers/121247-how-can-i-detect-and-remove-outliers-from-a-large-dataset
  
  global madRatioThreshold;

  % Get the median of those values.
  medianValue = nanmedian(pixelsInBlock(:));

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

  % Assign this to our output argument
  %output = MAD_Value;
  output = itIsAnOutlier;
  %output = [MAD_Value 255 * uint8(itIsAnOutlier)];
  %output = uint16(MAD_Value) + uint16(bitshift(itIsAnOutlier, 8));
