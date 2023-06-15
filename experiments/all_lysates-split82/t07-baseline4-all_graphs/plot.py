import matplotlib.pyplot as plt
import numpy as np

plot_freq = 50

########################################################################
# plot training history
########################################################################
# epoch,train_loss,valid_loss,train_pcc,valid_pcc,train_rmse,valid_rmse,
# train_mae,valid_mae,train_mse,valid_mse,train_r2,valid_r2
m = np.loadtxt(
	'training_history.csv', dtype=np.float_, delimiter=','
).reshape((-1,13))
n_epochs = m.shape[0]

plt.plot(m[:,0], m[:,1], label='train')
plt.plot(m[:,0], m[:,2], label='valid')
plt.title('loss (mean squared error)')
plt.xlabel('epoch')
plt.ylabel('mse (dimensionless)')
plt.xlim(-n_epochs//100, n_epochs+n_epochs//100)
plt.ylim(-0.15,3.15)
plt.legend()
plt.grid()
plt.savefig('training_history-loss_mse.png', dpi=300, bbox_inches='tight')
plt.close()

plt.plot(m[:,0], m[:,3], label='train')
plt.plot(m[:,0], m[:,4], label='valid')
plt.title('Pearson correlation coefficient')
plt.xlabel('epoch')
plt.ylabel('pcc')
plt.xlim(-n_epochs//100, n_epochs+n_epochs//100)
plt.ylim(-0.05,1.05)
plt.legend()
plt.grid()
plt.savefig('training_history-pcc.png', dpi=300, bbox_inches='tight')
plt.close()

plt.plot(m[:,0], m[:,5], label='train')
plt.plot(m[:,0], m[:,6], label='valid')
plt.title('root-mean-squared error')
plt.xlabel('epoch')
plt.ylabel('rmse (Celsius)')
plt.xlim(-n_epochs//100, n_epochs+n_epochs//100)
plt.ylim(-0.5,10.5)
plt.legend()
plt.grid()
plt.savefig('training_history-rmse.png', dpi=300, bbox_inches='tight')
plt.close()

plt.plot(m[:,0], m[:,7], label='train')
plt.plot(m[:,0], m[:,8], label='valid')
plt.title('mean absolute error')
plt.xlabel('epoch')
plt.ylabel('mae (Celsius)')
plt.xlim(-n_epochs//100, n_epochs+n_epochs//100)
plt.ylim(-0.5,10.5)
plt.legend()
plt.grid()
plt.savefig('training_history-mae.png', dpi=300, bbox_inches='tight')
plt.close()

plt.plot(m[:,0], m[:,9], label='train')
plt.plot(m[:,0], m[:,10], label='valid')
plt.title('mean squared error')
plt.xlabel('epoch')
plt.ylabel(r'mse (Celsius$^2$)')
plt.xlim(-n_epochs//100, n_epochs+n_epochs//100)
plt.ylim(-2.5,52.5)
plt.legend()
plt.grid()
plt.savefig('training_history-mse.png', dpi=300, bbox_inches='tight')
plt.close()

plt.plot(m[:,0], m[:,11], label='train')
plt.plot(m[:,0], m[:,12], label='valid')
plt.title(r'$r^2$')
plt.xlabel('epoch')
plt.ylabel(r'$r^2$')
plt.xlim(-n_epochs//100, n_epochs+n_epochs//100)
plt.ylim(-1.05,1.05)
plt.legend()
plt.grid()
plt.savefig('training_history-r2.png', dpi=300, bbox_inches='tight')
plt.close()

########################################################################
# plot true vs pred
########################################################################

# validation dataset
pred_file = 'predicted_values-valid_set.csv'
pred = np.loadtxt(pred_file, dtype=np.float_, delimiter=',')
with open(pred_file, 'r') as f:
	header = f.readline().replace('# ', '')
	true = f.readline().replace('# true_labels,', '')
true = np.array(true.split(','), dtype=np.float_)

for epoch in [1] + list(range(plot_freq, pred.shape[0]+1, plot_freq)):
	plt.scatter(true, pred[epoch-1,1:], marker='x', s=1, alpha=0.8, zorder=3)
	plt.plot(np.linspace(30,95), np.linspace(30,95), '--', c='k', alpha=0.3, zorder=1)
	plt.title('validation dataset\n' + rf'$r^2$={m[epoch-1,12]:.4f}')
	plt.xlabel('true')
	plt.ylabel('predicted')
	plt.grid()
	plt.gca().set_aspect('equal')
	plt.savefig(f'true_vs_predicted-valid_set-ep{epoch}.png', dpi=300, bbox_inches='tight')
	plt.close()

# training dataset
pred_file = 'predicted_values-train_set.csv'
pred = np.loadtxt(pred_file, dtype=np.float_, delimiter=',')
with open(pred_file, 'r') as f:
	header = f.readline().replace('# ', '')
	true = f.readline().replace('# true_labels,', '')
true = np.array(true.split(','), dtype=np.float_)

for epoch in [1] + list(range(plot_freq, pred.shape[0]+1, plot_freq)):
	plt.scatter(true, pred[epoch-1,1:], marker='x', s=1, alpha=0.8, zorder=3)
	plt.plot(np.linspace(30,95), np.linspace(30,95), '--', c='k', alpha=0.3, zorder=1)
	plt.title('training dataset\n' + rf'$r^2$={m[epoch-1,11]:.4f}')
	plt.xlabel('true')
	plt.ylabel('predicted')
	plt.grid()
	plt.gca().set_aspect('equal')
	plt.savefig(f'true_vs_predicted-train_set-ep{epoch}.png', dpi=300, bbox_inches='tight')
	plt.close()
