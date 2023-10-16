%% --------------------------------------------------------------------
%  DEBUG: quicksave/quickload results
% --------------------------------------------------------------------

load('full_mem_dump.mat')
quiverX = 3;

% Plot velocity magnitude
h1 = figure(1); clf
imshow(VM.magn, [0 max(VM.magn(:))], ...
       'InitialMagnification', 'fit', 'Colormap', jet(256))
title(['IW size = ' num2str(IW_size) ...
       ', overlap = ' num2str(IW_OVERLAP)])
axis on; axis tight; hold on
h1_ax = gca;

% Plot quiver map
h2 = figure(2); clf
set(gcf, 'Position', [2580, -225, 1280, 900])
quiver(VM.x, VM.y, VM.dx * quiverX, VM.dy * quiverX, ...
       0, 'w', 'LineWidth', 1.5)
set(gca, 'Position', [0.06 0.05 .9 .91], 'Color', [.2 .2 .8], 'YDir', 'reverse')
axis on; axis tight; hold on
xlim([1 img_w]); ylim([1 img_h])
h2_ax = gca;

%% --------------------------------------------------------------------
%  Detect bad vectors
% --------------------------------------------------------------------
if 0
% Filter vectors based on median absolute deviation of the magnitude
outlier_threshold = 1;  % 2
c = (VM.magn(:) - nanmedian(VM.magn(:))) / mad(VM.magn(:));
iReplaced = find(c > (nanmean(abs(c)) + ...
                      nanstd(abs(c)) * outlier_threshold));

% Plot bad vector indicators
[i, j] = ind2sub([IW_grid_A.nIWs_y, IW_grid_A.nIWs_x], iReplaced);
plot(h1_ax, j, i, 'xk', 'MarkerSize', 6, 'LineWidth', 2)
quiver(h2_ax, VM.x(iReplaced), VM.y(iReplaced), ...
       VM.dx(iReplaced) * quiverX, ...
       VM.dy(iReplaced) * quiverX, ...
       0, 'k', 'LineWidth', 2)





% Filter vectors based on the angle
tmp_grad = diff(unwrap(VM.angle));                  % column wise
%tmp_grad = [tmp_grad(1, :); tmp_grad];
tmp_grad = [tmp_grad; tmp_grad(end, :)];
iRep1 = find(tmp_grad(:) > 0.7 & c > -1);

tmp_grad = diff(unwrap(VM.angle, [], 2), [], 2);    % row wise
%tmp_grad = [tmp_grad(:, 1) tmp_grad];
tmp_grad = [tmp_grad tmp_grad(:, end)];
jRep2 = find(tmp_grad(:) > 0.7 & c > -1);
jRep2 = [];

% Plot bad vector indicators
[i, j] = ind2sub([IW_grid_A.nIWs_y, IW_grid_A.nIWs_x], unique([iRep1; jRep2]));
plot(h1_ax, j, i, 'xr', 'MarkerSize', 6, 'LineWidth', 2)
quiver(h2_ax, VM.x([iRep1; jRep2]), VM.y([iRep1; jRep2]), ...
       VM.dx([iRep1; jRep2]) * quiverX, ...
       VM.dy([iRep1; jRep2]) * quiverX, ...
       0, 'r', 'LineWidth', 2)

iReplaced = unique([iRep1; jRep2; iReplaced]);
nReplaced = length(iReplaced);




VM_filt = VM;
VM_filt.descr = 'filtered';
VM_filt.dx(iReplaced)   = nan;
VM_filt.dy(iReplaced)   = nan;
VM_filt.magn(iReplaced) = nan;
VM_filt.iReplaced       = iReplaced;
VM_filt.nReplaced       = nReplaced;

fprintf('filtered vectors: %i\n', nReplaced)
%disp('')
%pause
end

%% --------------------------------------------------------------------
%  Plot bad vectors, LP filter angle
% --------------------------------------------------------------------
h3 = figure(3);

angleThreshold = 30;  % [degrees]

%for iCol = 9:9
for iCol = 1:size(VM.x, 2)
  y = unwrap(VM.angle(:, iCol))/pi*180;

  % Split vector according to nans
  cut = isnan(y);
  id = cumsum(cut) + 1;
  mask = cut==0;
  out = accumarray(id(mask), y(mask), [], @(x) {x});
  offsets = find(cut == 1);

  for iSeg = 1:length(out)
    y = out{iSeg};

    if length(y) < 5
      % Do not perform outlier detection when segment is too short
      continue
    end

    if iSeg == 1
      y_offset = 0;
    else
      y_offset = offsets(iSeg - 1);
    end

    y_tiled = [flipud(y); y; flipud(y)];

    a = .5;
    y_LP = filtfilt(a, [1 a - 1], y_tiled);
    y_LP = y_LP(length(y) + 1:length(y) * 2);

    iR1 = find(abs((y - y_LP)) > angleThreshold);
    iR1 = iR1 + y_offset;
    iR1 = sub2ind(size(VM.x), iR1, ones(size(iR1)) * iCol);

    clf(h3)
    plot(y   , 'x-k', 'Linewidth', 2); hold on
    plot(y_LP, 'ob' , 'Linewidth', 2)
    plot(abs(y - y_LP), 'x-g', 'Linewidth', 2)
    plot(xlim, [angleThreshold angleThreshold], '-k')
    title(['column = ' num2str(iCol) ', px\_x = ' num2str(VM.x(1, iCol))])
    xlabel('px\_y')
    drawnow

    quiver(h2_ax, VM.x(iR1), VM.y(iR1), ...
           VM.dx(iR1) * quiverX, ...
           VM.dy(iR1) * quiverX, ...
           0, 'Color', 'r', 'LineWidth', 2)   % [237 177 32]/255
    %pause
  end
end

%% --------------------------------------------------------------------
%  Plot bad vectors, LP filter magnitude
% --------------------------------------------------------------------

magnDiffThreshold = 4;  % [degrees]

%for iCol = 9:9
for iCol = 1:size(VM.x, 2)
  y = VM.magn(:, iCol);

  % Split vector according to nans
  cut = isnan(y);
  id = cumsum(cut) + 1;
  mask = cut==0;
  out = accumarray(id(mask), y(mask), [], @(x) {x});
  offsets = find(cut == 1);

  for iSeg = 1:length(out)
    y = out{iSeg};

    if length(y) < 5
      % Do not perform outlier detection when segment is too short
      continue
    end

    if iSeg == 1
      y_offset = 0;
    else
      y_offset = offsets(iSeg - 1);
    end

    y_tiled = [flipud(y); y; flipud(y)];

    a = .5;
    y_LP = filtfilt(a, [1 a - 1], y_tiled);
    y_LP = y_LP(length(y) + 1:length(y) * 2);

    iR1 = find(abs((y - y_LP)) > magnDiffThreshold);
    iR1 = iR1 + y_offset;
    iR1 = sub2ind(size(VM.x), iR1, ones(size(iR1)) * iCol);

    clf(h3)
    plot(y   , 'x-k', 'Linewidth', 2); hold on
    plot(y_LP, 'ob' , 'Linewidth', 2)
    plot(abs(y - y_LP), 'x-g', 'Linewidth', 2)
    plot(xlim, [magnDiffThreshold magnDiffThreshold], '-k')
    title(['column = ' num2str(iCol) ', px\_x = ' num2str(VM.x(1, iCol))])
    xlabel('px\_y')
    drawnow

    quiver(h2_ax, VM.x(iR1), VM.y(iR1), ...
           VM.dx(iR1) * quiverX, ...
           VM.dy(iR1) * quiverX, ...
           0, 'Color', 'k', 'LineWidth', 2)   % [237 177 32]/255
    %pause
  end
end
