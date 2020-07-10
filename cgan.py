import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.utils as vutils
from cifar100 import get_cifar100

num_gpu = torch.cuda.device_count()
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar | lsun | mnist')
parser.add_argument('--dataroot', required=True, help='path to data')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='image size input')
parser.add_argument('--channels', type=int, default=3, help='number of channels')
parser.add_argument('--latentdim', type=int, default=100, help='size of latent vector')
parser.add_argument('--n_classes', type=int, default=100, help='number of classes in data set')
parser.add_argument('--epoch', type=int, default=200, help='number of epoch')
parser.add_argument('--lrate', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta', type=float, default=0.5, help='beta for adam optimizer')
parser.add_argument('--beta1', type=float, default=0.999, help='beta1 for adam optimizer')
parser.add_argument('--output', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--randomseed', type=int, help='seed')
 
opt = parser.parse_args()
print(opt)
img_shape = (opt.channels, opt.imageSize, opt.imageSize)
mean = [0.507, 0.487, 0.441]
std = [0.267, 0.256, 0.276]
def imshow(img):
	img = img / std + mean     # unnormalize
	npimg = img.detach().cpu().numpy()
	plt.imsave("gan.png",np.transpose(npimg, (1, 2, 0)))
	plt.show()
cuda = True if torch.cuda.is_available() else False 

os.makedirs(opt.output, exist_ok=True)

if opt.randomseed is None: 
	opt.randomseed = random.randint(1,10000)
random.seed(opt.randomseed)
torch.manual_seed(opt.randomseed)

# preprocessing for mnist, lsun, cifar10
if opt.dataset == 'mnist': 
	dataset = datasets.MNIST(root = opt.dataroot, train=True,download=True, 
		transform=transforms.Compose([transforms.Resize(opt.imageSize), 
			transforms.ToTensor(), 
			transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]))

elif opt.dataset == 'lsun': 
	dataset = datasets.LSUN(root = opt.dataroot, train=True,download=True, 
		transform=transforms.Compose([transforms.Resize(opt.imageSize), 
			transforms.CenterCrop(opt.imageSize),
			transforms.ToTensor(), 
			transforms.Normalize((0.5,), (0.5,))]))

elif opt.dataset == 'cifar':

	# dataset = datasets.CIFAR10(root = opt.dataroot, train=True,download=True,
	t=transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize(opt.imageSize),
	                                transforms.ToTensor(),
	                                transforms.Normalize(mean=mean, std=std)])
	dataset, train_unlabeled_dataset, _, test_data = get_cifar100('/cs/dataset/CIFAR/', n_labeled=50,
	                                                                            n_unlabled=1,transform_train=t)



assert dataset 
dataloader = torch.utils.data.DataLoader(dataset, batch_size = opt.batchSize, shuffle=True)

def check_folder(log_dir):
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	return log_dir
# building generator
class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.label_embed = nn.Embedding(opt.n_classes, opt.n_classes)
		self.depth=128

		def init(input, output, normalize=True):
			layers = [nn.Linear(input, output)]
			if normalize:
				layers.append(nn.BatchNorm1d(output, 0.8))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers

		self.generator = nn.Sequential(

			*init(opt.latentdim+opt.n_classes, self.depth),
			*init(self.depth, self.depth*2),
			*init(self.depth*2, self.depth*4),
			*init(self.depth*4, self.depth*8),
            nn.Linear(self.depth * 8, int(np.prod(img_shape))),
            nn.Tanh()

			)

	# torchcat needs to combine tensors
	def forward(self, noise, labels):
		gen_input = torch.cat((self.label_embed(labels), noise), -1)
		img = self.generator(gen_input)
		img = img.view(img.size(0), *img_shape)
		return img


class Discriminator(nn.Module): 
	def __init__(self): 
		super(Discriminator, self).__init__()
		self.label_embed1 = nn.Embedding(opt.n_classes, opt.n_classes)
		self.dropout = 0.4 
		self.depth = 512

		def init(input, output, normalize=True): 
			layers = [nn.Linear(input, output)]
			if normalize: 
				layers.append(nn.Dropout(self.dropout))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers 

		self.discriminator = nn.Sequential(
			*init(opt.n_classes+int(np.prod(img_shape)), self.depth, normalize=False),
			*init(self.depth, self.depth), 
			*init(self.depth, self.depth),
			nn.Linear(self.depth, 1),
			nn.Sigmoid()
			)

	def forward(self, img, labels): 
		imgs = img.view(img.size(0),-1)
		inpu = torch.cat((imgs, self.label_embed1(labels)), -1)
		validity = self.discriminator(inpu)
		return validity 

# weight initialization
def init_weights(m): 
	if type(m)==nn.Linear:
		torch.nn.init.xavier_uniform_(m.weight)
		m.bias.data.fill_(0.01)
	

# Building generator 
generator = Generator()
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lrate, betas=(opt.beta, opt.beta1))

# Building discriminator  
discriminator = Discriminator()
discriminator.apply(init_weights)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.lrate, betas=(opt.beta, opt.beta1))

# Loss functions 
a_loss = torch.nn.BCELoss()

# Labels 
real_label = 0.9
fake_label = 0.0


if cuda: 
	generator.cuda()
	discriminator.cuda()
	a_loss.cuda()
if num_gpu > 1:
	print(f"=>Using {num_gpu} GPUs!")
	generator.cuda()
	discriminator.cuda()
	a_loss.cuda()
	generator = nn.DataParallel(generator, device_ids=list(range(num_gpu)))
	discriminator = nn.DataParallel(discriminator, device_ids=list(range(num_gpu)))
	# a_loss.DataParallel(a_loss.cuda(), device_ids=list(range(num_gpu)))


def load_model(epoch,generator,discriminator):
	ckpt_g = torch.load(f'{os.getcwd()}/{opt.output}/generator_epoch_{epoch}.pth')
	generator.load_state_dict(ckpt_g)
	ckpt_d = torch.load(f'{os.getcwd()}/{opt.output}/discriminator_epoch_{epoch}.pth')
	discriminator.load_state_dict(ckpt_d)
	return generator, discriminator


#resume
try:
	epoch = 5000
	generator, discriminator= load_model(epoch,generator,discriminator)

except:
	generator, discriminator= load_model(1600,generator,discriminator)
	# training
	for epoch in range(1600,opt.epoch):
		for i, (imgs, labels) in enumerate(dataloader):
			batch_size = imgs.shape[0]

			# convert img, labels into proper form
			imgs=imgs.float().cuda()
			labels=labels.long().cuda()
			# label=torch.full((batch_size,), real_label, requires_grad=False).cuda()
			# creating real and fake tensors of labels
			reall = torch.FloatTensor(batch_size, 1).fill_(1.0).cuda()
			f_label = torch.FloatTensor(batch_size, 1).fill_(0.0).cuda()

			# initializing gradient
			gen_optimizer.zero_grad()
			d_optimizer.zero_grad()

			#### TRAINING GENERATOR ####
			# Feeding generator noise and labels
			noise = torch.randn(batch_size,  opt.latentdim).cuda()
			fixed_noise = torch.randn(batch_size,  opt.latentdim).cuda()

			gen_labels = torch.tensor(np.random.randint(0, opt.n_classes, batch_size)).cuda()

			gen_imgs = generator(noise, gen_labels)

			# Ability for discriminator to discern the real v generated images
			validity = discriminator(gen_imgs, gen_labels)

			# Generative loss function
			g_loss = a_loss(validity.squeeze(), reall.squeeze())

			# Gradients
			g_loss.backward()
			gen_optimizer.step()

			#### TRAINING DISCRIMINTOR ####

			d_optimizer.zero_grad()

			# Loss for real images and labels
			validity_real = discriminator(imgs, labels)
			d_real_loss = a_loss(validity_real.squeeze(), reall.squeeze())

			# Loss for fake images and labels
			validity_fake = discriminator(gen_imgs.detach(), gen_labels)
			d_fake_loss = a_loss(validity_fake.squeeze(), f_label.squeeze())

			# Total discriminator loss
			d_loss = 0.5 * (d_fake_loss+d_real_loss)

			# calculates discriminator gradients
			d_loss.backward()
			d_optimizer.step()


		# if epoch%100 == 0:
		imshow(torchvision.utils.make_grid(gen_imgs))

		vutils.save_image(gen_imgs, '%s/real_samples.png' % opt.output, normalize=True)
		fake = generator(noise, gen_labels)
		vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (opt.output, epoch), normalize=True)
		torch.save(generator.state_dict(), '%s/generator_epoch_%g.pth' % (opt.output, epoch))
		torch.save(discriminator.state_dict(), '%s/discriminator_epoch_%d.pth' % (opt.output, epoch))
		# np.savez_compressed(file_name + ".npz", data=fake.detach().cpu().numpy())
		print("[Epoch: %d/%d]" "[D loss: %f]" "[G loss: %f]" % (epoch+1, opt.epoch, d_loss.item(), g_loss.item()))

# save images
current_imgs=[]
current_labels=[]
generated_dataset=[]
generated_labels=[]

result_dir=f"{os.getcwd()}/cgan_imgs/"
for label in range(opt.n_classes):
	tmp = check_folder(result_dir)
num_iter = 5000 // opt.n_classes
z_fixed = torch.randn(opt.batchSize,  opt.latentdim).cuda()
for label in range(opt.n_classes):
	zeros = np.zeros(opt.batchSize, dtype=np.int64)
	zeros += label
	gen_labels = torch.tensor(zeros).cuda()
	for _ in range(num_iter):
		# clean samples z fixed - czcc
		# tmp = torch.tensor(np.random.randint(0, opt.n_classes, opt.batchSize)).cuda()
		# gen_labels[np.arange(opt.batchSize), y] = 1
		samples = generator(noise=z_fixed, labels=gen_labels)
		current_imgs.append(samples.detach().cpu().numpy())  # storing generated images and label
		current_labels += [label] * opt.batchSize
		samples_viz=np.transpose(samples.detach().cpu().numpy(), (1, 2, 0))
		if _ == 0:
			vutils.save_image(samples_viz, f'{result_dir}/label2_{label}.png',normalize=True)
		del samples
		# 	save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
		# 	            check_folder(result_dir + '/' + model_dir) + '/' + model_name + '_type_czcc' + '_label_%d.png' % label)
	# np.savez_compressed(f"{result_dir}/{label}", data=current__imgs, labels=current_labels)

	generated_dataset += current_imgs
	generated_labels += current_labels
	current_imgs=[]
	current_labels=[]
print(f"data size={len(generated_dataset)}, labeles:{generated_labels}")
np.savez_compressed(f"{result_dir}/all", data=generated_dataset, labels=generated_labels)
# checkpoints
# >>> np.savez_compressed('/tmp/123', a=test_array, b=test_vector)
# >>> loaded = np.load('/tmp/123.npz')














