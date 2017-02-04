import sys, os
import numpy as np
import conrad as cr

def demo(basepath='.'):
	# case with 1 target, 3 organs at risk. 1000 voxels x 200 beams
	case = cr.Case()
	case.anatomy += cr.medicine.Structure(0, 'Tumor', True, A=np.random.rand(100,200), dose=40*cr.Gy)
	case.anatomy += cr.medicine.Structure(1, 'OAR1', False, A=0.2*np.random.rand(200,200))
	case.anatomy += cr.medicine.Structure(2, 'OAR2', False, A=0.4*np.random.rand(300,200))
	case.anatomy += cr.medicine.Structure(3, 'OAR3', False, A=0.35*np.random.rand(400,200))

	graphics = cr.CasePlotter(case)
	graphics.dvh_plot.autoset_series_colors(colormap='rainbow')
	pd = case.plotting_data(x=np.random.rand(200))

	# 1 (sub)plot for all structures
	graphics.set_display_groups(grouping='together')
	# - with legend attached to figure
	graphics.plot(pd, plotfile=os.path.join(basepath, 'test1a.pdf'))
	# - with legend attached to upper-right-most subplot (there's only one)
	graphics.plot(pd, legend='upper_right',
				  plotfile=os.path.join(basepath, 'test1b.pdf'))

	# 1 subplot per structure
	graphics.set_display_groups(grouping='separate')
	# plot with y-axes on left-most subplots (default)
	graphics.plot(pd, plotfile=os.path.join(basepath, 'test2a.pdf'),
				  self_title_subplots=True)
	# plot with y-axes on all subplots
	graphics.plot(pd, plotfile=os.path.join(basepath, 'test2b.pdf'),
				  self_title_subplots=True, minimal_axes=False)

	# 2 subplots: structures 0 & 1 on first panel, 2 & 3 on second panel
	graphics.set_display_groups('list', [(0,1), (2,3)])
	graphics.plot(pd, plotfile=os.path.join(basepath, 'test3.pdf'),
				  self_title_subplots=True)

	# plot virtual lines to be represented in legend, not rendered in figure
	graphics.set_display_groups('list', [(0,1), (2,3)])
	graphics.plot(pd, legend=False)
	# aesthetics of legend lines
	aes = [cr.visualization.plot.LineAesthetic() for i in range(2)]
	aes[0].marker = 'o'
	aes[0].fill = 'full'
	aes[1].marker = 's'
	aes[1].fill = 'none'
	graphics.dvh_plot.plot_virtual(['first', 'second'], aes)
	graphics.dvh_plot.save(os.path.join(basepath,'test4.pdf'))

	# reference data
	pd_ref = case.plotting_data(x=np.random.rand(200))


	graphics.set_display_groups(grouping='separate')
	# plot series 'pd' vs. reference data, separate plots, automatic layout
	graphics.plot_multi([pd], ['first'], reference_data=pd_ref,
						plotfile=os.path.join(basepath,'test5.pdf'))
	# plot series 'pd' vs. reference data, separate plots, horizontal layout
	graphics.plot_multi([pd], ['first'], reference_data=pd_ref,
						plotfile=os.path.join(basepath,'test5a.pdf'),
						layout='horizontal', minimal_axes=False)
	# plot reference data only, separate plots, horizontal layout
	graphics.plot_multi([], [], reference_data=pd_ref,
						plotfile=os.path.join(basepath, 'test5b.pdf'),
						layout='horizontal')


	# plot multiple runs vs. reference, vary marker styles (default)
	RUNS = 3
	data = [case.plotting_data(x=np.random.rand(200)) for i in range(RUNS)]
	names = ['run{}'.format(i) for i in range(RUNS)]
	graphics.plot_multi(data, names, reference_data=pd_ref,
						plotfile=os.path.join(basepath, 'test5c.pdf'),
						layout='horizontal', minimal_axes=False)

	# plot even more runs vs. reference, go deeper into marker styles variation
	RUNS = 6
	data = [case.plotting_data(x=np.random.rand(200)) for i in range(RUNS)]
	names = ['run{}'.format(i) for i in range(RUNS)]
	graphics.plot_multi(data, names, reference_data=pd_ref,
						plotfile=os.path.join(basepath, 'test5d.pdf'),
						layout='horizontal', minimal_axes=False)

	# plot multiple runs vs. reference, vary marker styles (default)
	RUNS = 3
	data = [case.plotting_data(x=np.random.rand(200)) for i in range(RUNS)]
	names = ['run{}'.format(i) for i in range(RUNS)]
	graphics.plot_multi(data, names, reference_data=pd_ref,
						plotfile=os.path.join(basepath, 'test5e.pdf'),
						layout='horizontal', minimal_axes=False,
						universal_marker='o')

	# plot runs vs. reference, don't plot prescription as vertical line,
	# don't vary markers. instead...
	RUNS = 4
	data = [case.plotting_data(x=np.random.rand(200)) for i in range(RUNS)]
	names = ['run{}'.format(i) for i in range(RUNS)]
	# ...vary line styles
	graphics.plot_multi(data, names, reference_data=pd_ref,
						plotfile=os.path.join(basepath, 'test5f.pdf'),
						layout='horizontal', minimal_axes=False,
						suppress_rx=True, vary_markers=False,
						vary_line_styles=True)
	# ...vary line weights and colors
	graphics.plot_multi(data, names, reference_data=pd_ref,
						plotfile=os.path.join(basepath, 'test5g.pdf'),
						layout='horizontal', minimal_axes=False,
						vary_markers=False, vary_line_weights=True,
						vary_line_colors=True)

def clean(basepath='.'):
	tags = ['1a', '1b', '2a', '2b', '3', '4', '5a', '5b', '5c', '5d',
			 '5e', '5f', '5g']
	for plot_ in map(lambda t: os.path.join(basepath, 'test' + t + '.pdf'), tags):
		if os.path.exists(plot_):
			os.remove(plot_)

if __name__ == '__main__':
	basepath = ''
	for arg in sys.argv:
		if '--path=' in arg:
			basepath = os.path.abspath(arg.replace('--path=', ''))
			break
	if basepath == '':
		basepath = os.path.abspath('.')

	if '--clean' in sys.argv:
		clean(basepath=basepath)
	else:
		demo(basepath=basepath)