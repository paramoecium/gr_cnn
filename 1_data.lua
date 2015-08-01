----------------------------------------------------------------------
-- It's a good idea to run this script with the interactive mode:
-- $ th -i 1_data.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to analyze/visualize the data you've just loaded.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides a normalization operator

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Gait Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
-- my utility functions
function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end
----------------------------------------------------------------------
train_file = 'alignedData_train.csv'
test_file = 'alignedData_test.csv'
trsize = -1
tesize = -1
signal_len = -1
----------------------------------------------------------------------
print '==> loading dataset'
-- Count number of rows and columns in file
instance_id = -1
for line in io.lines(train_file) do
   local l = line:split(',')
   if instance_id ~= l[1] then
      instance_id = l[1]
      signal_len = 0
   end
   signal_len = signal_len + 1
end
trsize = instance_id + 1

-- Read data from CSV to tensor
data = torch.Tensor(trsize,2,3,signal_len) -- accel*3, alpha*3
labels = torch.Tensor(trsize)

i = 1
instance_id = -1
for line in io.lines(train_file) do
   local l = line:split(',')
   if instance_id ~= l[1] + 1 then
      instance_id = l[1] + 1
      labels[instance_id] = l[2]
      i = 1
   end
   instance_id = l[1] + 1
   local value_id = l[1] + 1
   for key, val in ipairs(l) do
      repeat
         local col = key - 2
         if col < 1 then
            break
         elseif col < 4 then
            data[instance_id][1][col][i] = val
         else
            data[instance_id][2][col-3][i] = val
         end
      until true
   end
   i = i + 1
end


-- Note: the data, in X, is 4-d: the 1st dim indexes the samples, the 2nd
-- dim indexes the  channels (accel or alpha), and the last two dims index the
-- direction(height) and timestamp(width) of the samples.

trainData = {
   data = data,
   labels = labels,
   size = function() return trsize end
}

-- Finally we load the test data.
-- Count number of rows and columns in file
instance_id = -1
for line in io.lines(test_file) do
   local l = line:split(',')
   if instance_id ~= l[1] then
      instance_id = l[1]
      signal_len = 0
   end
   signal_len = signal_len + 1
end
tesize = instance_id + 1

-- Read data from CSV to tensor
data = torch.Tensor(tesize,2,3,signal_len) -- accel*3, alpha*3
labels = torch.Tensor(tesize)

i = 1
instance_id = -1
for line in io.lines(test_file) do
   local l = line:split(',')
   if instance_id ~= l[1] + 1 then
      instance_id = l[1] + 1
      labels[instance_id] = l[2]
      i = 1
   end
   instance_id = l[1] + 1
   local value_id = l[1] + 1
   for key, val in ipairs(l) do
      repeat
         local col = key - 2
         if col < 1 then
            break
         elseif col < 4 then
            data[instance_id][1][col][i] = val
         else
            data[instance_id][2][col-3][i] = val
         end
      until true
   end
   i = i + 1
end

testData = {
   data = data,
   labels = labels,
   size = function() return tesize end
}

----------------------------------------------------------------------
print '==> preprocessing data'

-- We now preprocess the data. Preprocessing is crucial
-- when applying pretty much any kind of machine learning algorithm.

-- For natural images, we use several intuitive tricks:
--   + images are mapped into YUV space, to separate luminance information
--     from color information
--   + channels is locally normalized, using a contrastive
--     normalization operator: for each neighborhood, defined by a Gaussian
--     kernel, the mean is suppressed, and the standard deviation is normalized
--     to one.
channels = {'accel','alpha'}
----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do
   trainMean = trainData.data[{ {},i }]:mean()
   trainStd = trainData.data[{ {},i }]:std()

   testMean = testData.data[{ {},i }]:mean()
   testStd = testData.data[{ {},i }]:std()

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..channel..'-channel, mean: ' .. testMean)
   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end

----------------------------------------------------------------------
print '==> visualizing data'

-- Visualization is quite easy, using itorch.image().

if opt.visualize then
   if itorch then
   first256Samples_y = trainData.data[{ {1,256},1 }]
   first256Samples_u = trainData.data[{ {1,256},2 }]
   first256Samples_v = trainData.data[{ {1,256},3 }]
   itorch.image(first256Samples_y)
   itorch.image(first256Samples_u)
   itorch.image(first256Samples_v)
   else
      print("For visualization, run this script in an itorch notebook")
   end
end
