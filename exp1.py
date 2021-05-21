from matplotlib import pyplot as plt
import numpy
import sympy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import tkinter
import os
try:
	os.mkdir('up/')
except:
	pass
try:
	os.mkdir('down/')
except:
	pass
try:
	os.mkdir('hand/')
except:
	pass
Times=0
locked=False
expry1n=None
exprz1n=None
exprstep1n=None
expry2n=None
exprz2n=None
exprstep2n=None
Galpha=None
Gwb=None
Gllaser=None
Ghlaser=None
Glaserx=None
Glasery=None
Glaserz=None
Glasernx=None
Glaserny=None
Glasernz=None
Glaserax=None
Glaseray=None
Glaseraz=None
Glasertx=None
Glaserty=None
Glasertz=None
Grangy=None
Grangz=None
Glamd=None
Gscreenx=None
Gscreeny=None
Gscreenz=None

def p(r):
	return 5*2.71828**(-0.505*(r-1)**2)

def fpzt(r,v,w):
	return max(v-r,min(v,w))

def updateF(x,Fapzt,Fppzt,Papzt):
	for i in range(M):
		Fapzt[i][0]=fpzt(i*dr,x,Fapzt[i][0])
	Fppzt=Fapzt*Papzt
	return Fapzt,Fppzt 

def  y(x,Fp,dr):
	return p0*x+sum(Fp)*dr

def F(r,x,Fa,n):
	if n==0:
		return fpzt(r,x[0],0)
	else:
		return fpzt(r,x[n],Fl[n-1])

p0=2.42
R=1
M=1000
dr=R/M
Fapzt=numpy.zeros((M,1))
Papzt=numpy.zeros((M,1))
for i in range(M):
	Papzt[i][0]+=p(i*dr)

Fppzt=Fapzt*Papzt

def pztD(V):
	global Fppzt
	global Fapzt
	D=y(V,Fppzt,dr)
	Fapzt,Fppzt=updateF(V,Fapzt,Fppzt,Papzt)
	return D[0]*0.00001

#45 1500 50 23 13
#30 3500 50 34 16
def fuckfinal(lights1x,lights1y,lights1z,lights1nx,lights1ny,lights1nz,lights1s,lights2x,lights2y,lights2z,lights2nx,lights2ny,lights2nz,lights2s,wb,lamd):
	lymax=int(numpy.max(lights1y))
	lymin=int(numpy.min(lights1y))
	lycpmax=int(numpy.max(lights2y))
	lycpmin=int(numpy.min(lights2y))

	lytmin=max(lymin,lycpmin)
	lytmax=min(lymax,lycpmax)
	shifty=1-lytmin
	rangy=lytmax+shifty+2
	scaley=50
	Rangy=rangy*scaley

	print(lymax,lymin,lycpmax,lycpmin,shifty,rangy,Rangy)
	lzmax=int(numpy.max(lights1z))
	lzmin=int(numpy.min(lights1z))
	lzcpmax=int(numpy.max(lights2z))
	lzcpmin=int(numpy.min(lights2z))

	lztmin=max(lzmin,lzcpmin)
	lztmax=min(lzmax,lzcpmax)
	shiftz=1-lztmin
	rangz=lztmax+shiftz+2
	scalez=50
	Rangz=rangz*scalez

	print(lzmax,lzmin,lzcpmax,lzcpmin,shiftz,rangz,Rangz)
	StRang=min(Rangy,Rangz)	
	
	print(StRang)
	amp=1
	final2R=numpy.zeros((StRang,StRang))
	final2I=numpy.zeros((StRang,StRang))

	for i in range(wb):
		for k in range(wb):
			lighty=lights1y[i][k]
			lightz=lights1z[i][k]
			step=lights1s[i][k]
			y=int((lighty+shifty)*scaley)
			z=int((lightz+shiftz)*scalez)
			if y<StRang and z<StRang and y>0 and z>0:
				final2R[y][z]+=numpy.cos(2*numpy.pi*step/lamd)*amp
				final2I[y][z]+=numpy.sin(2*numpy.pi*step/lamd)*amp
	for i in range(wb):
		for k in range(wb):
			lighty=lights2y[i][k]
			lightz=lights2z[i][k]
			step=lights2s[i][k]
			y=int((lighty+shifty)*scaley)
			z=int((lightz+shiftz)*scalez)
			if y<StRang and z<StRang and y>0 and z>0:
				final2R[y][z]+=numpy.cos(2*numpy.pi*step/lamd)*amp
				final2I[y][z]+=numpy.sin(2*numpy.pi*step/lamd)*amp
	
	final2M=final2R**2+final2I**2

	plt.imshow(final2M)
	plt.savefig('final2M.png')
	plt.close()
	return final2M

def fuckfinal2(rangy,rangz,screenx,screeny,screenz,lights1y,lights1z,lights1s,lights2y,lights2z,lights2s,wb,lamd):
	StRang=400 # -5 -> +5
	scaley=StRang/(2*rangy)
	scalez=StRang/(2*rangz)
	ymin=screeny-rangy
	ymax=screeny+rangy
	zmin=screenz-rangz
	zmax=screenz+rangz
	amp=1
	final2R=numpy.zeros((StRang,StRang))
	final2I=numpy.zeros((StRang,StRang))
	coef=2*numpy.pi/lamd
	lights1scoef=coef*lights1s
	lights2scoef=coef*lights2s
	stepc1=numpy.cos(lights1scoef)
	steps1=numpy.sin(lights1scoef)
	stepc2=numpy.cos(lights2scoef)
	steps2=numpy.sin(lights2scoef)
	for i in range(wb):
		for k in range(wb):
			lighty=lights1y[i][k]
			lightz=lights1z[i][k]
			if lighty<ymax and lightz<zmax and lighty>ymin and lightz>zmin:
				y=int((lighty-ymin)*scaley)
				z=int((lightz-zmin)*scalez)
				final2R[y][z]+=stepc1[i][k]
				final2I[y][z]+=steps1[i][k]
	for i in range(wb):
		for k in range(wb):
			lighty=lights2y[i][k]
			lightz=lights2z[i][k]
			if lighty<ymax and lightz<zmax and lighty>ymin and lightz>zmin:
				y=int((lighty-ymin)*scaley)
				z=int((lightz-zmin)*scalez)
				final2R[y][z]+=stepc2[i][k]
				final2I[y][z]+=steps2[i][k]

	
	final2M=final2R**2+final2I**2
	#plt.imshow(final2M)
	#plt.savefig('final2M.png')
	#plt.close()
	return final2M
def Send_OutandGet_Big(alpha,wb,llaser,hlaser,laserx,lasery,laserz,lasernx,laserny,lasernz,laserax,laseray,laseraz,lasertx,laserty,lasertz):
	#激光射出
	O=numpy.ones((wb,wb))
	PI=numpy.zeros((wb,wb))
	PJ=numpy.zeros((wb,wb))
	for i in range(wb):
		for j in range(wb):
			PI[i][j]=i*llaser/wb
			PJ[i][j]=j*hlaser/wb
	lightsx=(laserx+laserax*llaser/2+lasertx*hlaser/2)*O-laserax*PI-lasertx*PJ
	lightsy=(lasery+laseray*llaser/2+laserty*hlaser/2)*O-laseray*PI-laserty*PJ
	lightsz=(laserz+laseraz*llaser/2+lasertz*hlaser/2)*O-laseraz*PI-lasertz*PJ
	lightsnx=(lasernx-alpha*laserx)*O+alpha*(lightsx)
	lightsny=(laserny-alpha*lasery)*O+alpha*(lightsy)
	lightsnz=(lasernz-alpha*laserz)*O+alpha*(lightsz)
	norm=(lightsnx*lightsnx+lightsny*lightsny+lightsnz*lightsnz)**(0.5)
	lightsnx/=norm
	lightsny/=norm
	lightsnz/=norm
	return lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz

def LaserToSplit(x1n,y1n,z1n,nx1n,ny1n,nz1n,step1n,x2n,y2n,z2n,nx2n,ny2n,nz2n,step2n,splitn2,splitnx,splitny,splitnz):
	k=(splitn2-x1n*splitnx-y1n*splitny-z1n*splitnz)/(nx1n*splitnx+ny1n*splitny+nz1n*splitnz)
	x1n=x1n+k*nx1n
	y1n=y1n+k*ny1n
	z1n=z1n+k*nz1n
	step1n+=k

	n1dn2=nx1n*splitnx+ny1n*splitny+nz1n*splitnz
	x2n=x1n
	y2n=y1n
	z2n=z1n
	nx2n=nx1n-2*n1dn2*splitnx
	ny2n=ny1n-2*n1dn2*splitny
	nz2n=nz1n-2*n1dn2*splitnz
	step2n+=k

	return x1n,y1n,z1n,nx1n,ny1n,nz1n,step1n,x2n,y2n,z2n,nx2n,ny2n,nz2n,step2n
def betweeninv(xn,yn,zn,nxn,nyn,nzn,stepn,invn2,invnx,invny,invnz):
	k=(invn2-xn*invnx-yn*invny-zn*invnz)/(nxn*invnx+nyn*invny+nzn*invnz)
	xn+=k*nxn
	yn+=k*nyn
	zn+=k*nzn
	stepn+=k

	n1dn2=nxn*invnx+nyn*invny+nzn*invnz
	nxn+=-2*n1dn2*invnx
	nyn+=-2*n1dn2*invny
	nzn+=-2*n1dn2*invnz

	return xn,yn,zn,nxn,nyn,nzn,stepn


def _quit():
    root.quit()

def _plot():
	lamd=float(iptlamd.get())
	N=float(iptN.get())
	V=float(iptV.get())
	hlaser=0.2
	wlaser=0.2
	llaser=0.2
	laserx=float(iptlaserx.get())
	lasery=float(iptlasery.get())
	laserz=float(iptlaserz.get())
	lasernx=float(iptlasernx.get())
	laserny=float(iptlaserny.get())
	lasernz=float(iptlasernz.get())
	lasertx=float(iptlasertx.get())
	laserty=float(iptlaserty.get())
	lasertz=float(iptlasertz.get())
	laserax=float(iptlaserax.get())
	laseray=float(iptlaseray.get())
	laseraz=float(iptlaseraz.get())
	splitx=float(iptsplitx.get())
	splity=float(iptsplity.get())
	splitz=float(iptsplitz.get())
	splitnx=float(iptsplitnx.get())
	splitny=float(iptsplitny.get())
	splitnz=float(iptsplitnz.get())
	splitn2=splitx*splitnx+splity*splitny+splitz*splitnz
	reflectx=float(iptreflectx.get())
	reflecty=float(iptreflecty.get())
	reflectz=float(iptreflectz.get())
	reflectnx=float(iptreflectnx.get())
	reflectny=float(iptreflectny.get())
	reflectnz=float(iptreflectnz.get())
	reflectn2=reflectx*reflectnx+reflecty*reflectny+reflectz*reflectnz
	pztx=float(iptpztx.get())
	pzty=float(iptpzty.get())
	pztz=float(iptpztz.get())
	pztnx=float(iptpztnx.get())
	pztny=float(iptpztny.get())
	pztnz=float(iptpztnz.get())
	Dp=pztD(V)
	pztx+=pztnx*Dp
	pzty+=pztny*Dp
	pztz+=pztnz*Dp
	pztn2=pztx*pztnx+pzty*pztny+pztz*pztnz
	screenx=float(iptscreenx.get())
	screeny=float(iptscreeny.get())
	screenz=float(iptscreenz.get())
	screennx=float(iptscreennx.get())
	screenny=float(iptscreenny.get())
	screennz=float(iptscreennz.get())
	screenn2=screenx*screennx+screeny*screenny+screenz*screennz
	alpha=float(iptalpha.get())
	wb=int(N**0.5)
	a.clear()
	rangy=5
	rangz=5
	wb=int(N**0.5)

	x0 = sympy.Symbol('x')
	y0 = sympy.Symbol('y')
	z0 = sympy.Symbol('z')
	nx0= sympy.Symbol('nx')
	ny0= sympy.Symbol('ny')
	nz0= sympy.Symbol('nz')
	
	x1n=x0
	y1n=y0
	z1n=z0
	nx1n=nx0
	ny1n=ny0
	nz1n=nz0
	step1n=0

	x2n=x0
	y2n=y0
	z2n=z0
	nx2n=nx0
	ny2n=ny0
	nz2n=nz0
	step2n=0

	x1n,y1n,z1n,nx1n,ny1n,nz1n,step1n,x2n,y2n,z2n,nx2n,ny2n,nz2n,step2n=LaserToSplit(x1n,y1n,z1n,nx1n,ny1n,nz1n,step1n,x2n,y2n,z2n,nx2n,ny2n,nz2n,step2n,splitn2,splitnx,splitny,splitnz)
	#print('laser to split')

	x1n,y1n,z1n,nx1n,ny1n,nz1n,step1n=betweeninv(x1n,y1n,z1n,nx1n,ny1n,nz1n,step1n,pztn2,pztnx,pztny,pztnz)
	#print('split to pzt')

	x2n,y2n,z2n,nx2n,ny2n,nz2n,step2n=betweeninv(x2n,y2n,z2n,nx2n,ny2n,nz2n,step2n,reflectn2,reflectnx,reflectny,reflectnz)
	#print('split to reflect')

	x1n,y1n,z1n,nx1n,ny1n,nz1n,step1n=betweeninv(x1n,y1n,z1n,nx1n,ny1n,nz1n,step1n,splitn2,splitnx,splitny,splitnz)
	#print('pzt to split')

	x1n,y1n,z1n,nx1n,ny1n,nz1n,step1n=betweeninv(x1n,y1n,z1n,nx1n,ny1n,nz1n,step1n,screenn2,screennx,screenny,screennz)
	#print('pzt-split to screen')

	x2n,y2n,z2n,nx2n,ny2n,nz2n,step2n=betweeninv(x2n,y2n,z2n,nx2n,ny2n,nz2n,step2n,screenn2,screennx,screenny,screennz)
	#print('reflect to screen')

	lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz=Send_OutandGet_Big(alpha,wb,llaser,hlaser,laserx,lasery,laserz,lasernx,laserny,lasernz,laserax,laseray,laseraz,lasertx,laserty,lasertz)
	#print('light inited')

	f1y=sympy.lambdify(('x','y','z','nx','ny','nz'), y1n, "numpy")
	f1z=sympy.lambdify(('x','y','z','nx','ny','nz'), z1n, "numpy")
	f1s=sympy.lambdify(('x','y','z','nx','ny','nz'), step1n, "numpy")
	f2y=sympy.lambdify(('x','y','z','nx','ny','nz'), y2n, "numpy")
	f2z=sympy.lambdify(('x','y','z','nx','ny','nz'), z2n, "numpy")
	f2s=sympy.lambdify(('x','y','z','nx','ny','nz'), step2n, "numpy")

	lights1y=f1y(lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz)
	lights1z=f1z(lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz)
	lights1s=f1s(lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz)
	#print('1 done')

	lights2y=f2y(lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz)
	lights2z=f2z(lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz)
	lights2s=f2s(lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz)
	#print('2 done')

	c=fuckfinal2(rangy,rangz,screenx,screeny,screenz,lights1y,lights1z,lights1s,lights2y,lights2z,lights2s,wb,lamd)
	a.imshow(c)

	canvas.draw()

def _plot2():
	V=float(iptV.get())
	Dp=pztD(V)
	y1n=expry1n.subs({'dp':Dp})
	z1n=exprz1n.subs({'dp':Dp})
	step1n=exprstep1n.subs({'dp':Dp})
	y2n=expry2n.subs({'dp':Dp})
	z2n=exprz2n.subs({'dp':Dp})
	step2n=exprstep2n.subs({'dp':Dp})

	f1y=sympy.lambdify(('x','y','z','nx','ny','nz'), y1n, "numpy")
	f1z=sympy.lambdify(('x','y','z','nx','ny','nz'), z1n, "numpy")
	f1s=sympy.lambdify(('x','y','z','nx','ny','nz'), step1n, "numpy")
	f2y=sympy.lambdify(('x','y','z','nx','ny','nz'), y2n, "numpy")
	f2z=sympy.lambdify(('x','y','z','nx','ny','nz'), z2n, "numpy")
	f2s=sympy.lambdify(('x','y','z','nx','ny','nz'), step2n, "numpy")

	lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz=Send_OutandGet_Big(Galpha,Gwb,Gllaser,Ghlaser,Glaserx,Glasery,Glaserz,Glasernx,Glaserny,Glasernz,Glaserax,Glaseray,Glaseraz,Glasertx,Glaserty,Glasertz)

	lights1y=f1y(lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz)
	lights1z=f1z(lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz)
	lights1s=f1s(lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz)
	#print('1 done')

	lights2y=f2y(lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz)
	lights2z=f2z(lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz)
	lights2s=f2s(lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz)
	#print('2 done')

	c=fuckfinal2(Grangy,Grangz,Gscreenx,Gscreeny,Gscreenz,lights1y,lights1z,lights1s,lights2y,lights2z,lights2s,Gwb,Glamd)
	a.imshow(c)
	plt.imshow(c)
	global Times
	Times+=1
	plt.savefig('hand/'+str(Times)+' V='+str(V)+'.png')
	labout.configure(text='手动实验采集完毕,陶瓷两端电压为 '+str(V)+' V')
	canvas.draw()

def _plot4(up,V):
	Dp=pztD(V)
	labout.configure(text='陶瓷两端电压： '+str(V)+' V')
	y1n=expry1n.subs({'dp':Dp})
	z1n=exprz1n.subs({'dp':Dp})
	step1n=exprstep1n.subs({'dp':Dp})
	y2n=expry2n.subs({'dp':Dp})
	z2n=exprz2n.subs({'dp':Dp})
	step2n=exprstep2n.subs({'dp':Dp})

	f1y=sympy.lambdify(('x','y','z','nx','ny','nz'), y1n, "numpy")
	f1z=sympy.lambdify(('x','y','z','nx','ny','nz'), z1n, "numpy")
	f1s=sympy.lambdify(('x','y','z','nx','ny','nz'), step1n, "numpy")
	f2y=sympy.lambdify(('x','y','z','nx','ny','nz'), y2n, "numpy")
	f2z=sympy.lambdify(('x','y','z','nx','ny','nz'), z2n, "numpy")
	f2s=sympy.lambdify(('x','y','z','nx','ny','nz'), step2n, "numpy")

	lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz=Send_OutandGet_Big(Galpha,Gwb,Gllaser,Ghlaser,Glaserx,Glasery,Glaserz,Glasernx,Glaserny,Glasernz,Glaserax,Glaseray,Glaseraz,Glasertx,Glaserty,Glasertz)

	lights1y=f1y(lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz)
	lights1z=f1z(lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz)
	lights1s=f1s(lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz)
	#print('1 done')

	lights2y=f2y(lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz)
	lights2z=f2z(lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz)
	lights2s=f2s(lightsx,lightsy,lightsz,lightsnx,lightsny,lightsnz)
	#print('2 done')

	c=fuckfinal2(Grangy,Grangz,Gscreenx,Gscreeny,Gscreenz,lights1y,lights1z,lights1s,lights2y,lights2z,lights2s,Gwb,Glamd)
	#a.imshow(c)
	plt.imshow(c)
	if up==True:
		plt.savefig('up/'+str(V)+'.png')
	else:
		plt.savefig('down/'+str(V)+'.png')
	#canvas.draw()

def _plot3():
	v1=float(iptv1.get())
	v2=float(iptv2.get())
	vd=float(iptvd.get())
	vlup=[]
	vldown=[]
	vn=v1
	while vn<v2:
		vlup.append(int(vn*1000)/1000)
		vn+=vd
	vn-=vd
	while vn>v1:
		vldown.append(int(vn*1000)/1000)
		vn-=vd
	for v in vlup:
		_plot4(True,v)
	for v in vldown:
		_plot4(False,v)
	labout.configure(text='自动实验采集完毕')

root = tkinter.Tk()
w, h = root.maxsize()
root.geometry("{}x{}".format(w, h))

#root.attributes("-fullscreen", True)
root.title("模拟迈克尔逊干涉仪")

f = Figure(figsize=(5, 5), dpi=100)
a = f.add_subplot(111)  # 添加子图:1行1列第1个
canvas = FigureCanvasTkAgg(f, master=root)
canvas.get_tk_widget().pack(side=tkinter.TOP,fill=tkinter.BOTH,expand=tkinter.YES)
toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas._tkcanvas.place(relx=0.40,rely=0.7,anchor='sw')
#canvas._tkcanvas.pack(side=tkinter.TOP,fill=tkinter.BOTH,expand=tkinter.YES)

btnplot = tkinter.Button(master=root, text="开始模拟光屏图样", command=_plot)
btnplot.place(relx=0.0, rely=0.68, anchor='nw')
btnplot2=tkinter.Button(master=root,text='开始手动实验',command=_plot2)
btnplot2.place_forget()
btnplot3=tkinter.Button(master=root,text='开始自动实验',command=_plot3)
btnplot3.place_forget()
btnquit = tkinter.Button(master=root, text="退出", command=_quit)
btnquit.place(relx=0.12, rely=0.68, anchor='nw')
infotext='本实验是探究压电陶瓷静态滞回现象的虚拟实验.\n\n首先在面板左边调节各项实验参数,长度单位均为\n厘米,调节过程中,可以点击"开始模拟光屏图样",\n获得光屏处图样.\n\n调节完毕后,点击"锁定参数",此后参数不再可调节,\n若想调节参数必须点击"解锁参数".\n\n若想手动进行实验,输入陶瓷两端电压值,点击"手动实验",\n测得结果以电压值为名放在hand文件夹下.\n\n若想自动进行实验,输入最小电压值,最大电压值和电压步长,\n点击"自动实验",则自动从最小电压测到最大电压,\n再从最大电压测回最小电压.\n分别存放在up文件夹和down文件夹下.'
labinfo=tkinter.Label(root,text=infotext,wraplength=0,width=0,height=0)
labinfo.place(relx=0.75,rely=0,anchor='nw')

labout=tkinter.Label(master=root,text='')
labout.place(relx=0.5,rely=0.7 ,anchor='nw')

def getlabipt(name,val,x,y):
	lab=tkinter.Label(root,text=name)
	lab.place(relx=x,rely=y,anchor='nw')
	ipt=tkinter.Entry(root,highlightcolor='red',highlightthickness=1,textvariable=tkinter.StringVar(value=val))
	ipt.place(relx=x+0.06,rely=y,anchor='nw',width=70,height=20)
	labipt=tkinter.Label(root,text=name)
	labipt.place(relx=x+0.06,rely=y,anchor='nw')
	labipt.place_forget()
	return lab,ipt,labipt

#激光器参数设置
lablamd,iptlamd,labiptlamd = getlabipt('激光波长','0.0000628',0.12,0.58)
labN,iptN,labiptN = getlabipt('模拟光线数','100000',0.24,0.58)
labV,iptV,labiptV = getlabipt('陶瓷加电压/V','0',0.0,0.72)
labV.place_forget()
iptV.place_forget()
lablasernx,iptlasernx,labiptlasernx = getlabipt('激光法向x','0',0.12,0.0)
lablaserny,iptlaserny,labiptlaserny = getlabipt('激光法向y','-1',0.12,0.04)
lablasernz,iptlasernz,labiptlasernz = getlabipt('激光法向z','0',0.12,0.08)
lablasertx,iptlasertx,labiptlasertx = getlabipt('激光主切向x','0',0.24,0.0)
lablaserty,iptlaserty,labiptlaserty = getlabipt('激光主切向y','0',0.24,0.04)
lablasertz,iptlasertz,labiptlasertz = getlabipt('激光主切向z','1',0.24,0.08)
lablaserax,iptlaserax,labiptlaserax = getlabipt('激光副切向x','-1',0.24,0.12)
lablaseray,iptlaseray,labiptlaseray = getlabipt('激光副切向y','0',0.24,0.16)
lablaseraz,iptlaseraz,labiptlaseraz = getlabipt('激光副切向z','0',0.24,0.20)
lablaserx,iptlaserx,labiptlaserx = getlabipt('激光位置x','0',0.12,0.12)
lablasery,iptlasery,labiptlasery = getlabipt('激光位置y','0',0.12,0.16)
lablaserz,iptlaserz,labiptlaserz = getlabipt('激光位置z','0',0.12,0.20)

#光屏参数设置
labscreennx,iptscreennx,labiptscreennx = getlabipt('光屏法向x','1',0.0,0.0)
labscreenny,iptscreenny,labiptscreenny = getlabipt('光屏法向y','0',0.0,0.04)
labscreennz,iptscreennz,labiptscreennz = getlabipt('光屏法向z','0',0.0,0.08)
labscreenx,iptscreenx,labiptscreenx = getlabipt('光屏位置x','-10',0.0,0.12)
labscreeny,iptscreeny,labiptscreeny = getlabipt('光屏位置y','-10',0.0,0.16)
labscreenz,iptscreenz,labiptscreenz = getlabipt('光屏位置z','0',0.0,0.20)

#分束镜参数设置
labsplitnx,iptsplitnx,labiptsplitnx = getlabipt('分束镜法向x','0.70710678',0.0,0.30)
labsplitny,iptsplitny,labiptsplitny = getlabipt('分束镜法向y','0.70710678',0.0,0.34)
labsplitnz,iptsplitnz,labiptsplitnz = getlabipt('分束镜法向z','0',0.0,0.38)
labsplitx,iptsplitx,labiptsplitx = getlabipt('分束镜位置x','0',0.0,0.42)
labsplity,iptsplity,labiptsplity = getlabipt('分束镜位置y','-10',0.0,0.46)
labsplitz,iptsplitz,labiptsplitz = getlabipt('分束镜位置z','0',0.0,0.50)
#反射镜参数设置
labreflectnx,iptreflectnx,labiptreflectnx = getlabipt('反射镜法向x','-1',0.12,0.30)
labreflectny,iptreflectny,labiptreflectny = getlabipt('反射镜法向y','0',0.12,0.34)
labreflectnz,iptreflectnz,labiptreflectnz = getlabipt('反射镜法向z','0',0.12,0.38)
labreflectx,iptreflectx,labiptreflectx = getlabipt('反射镜位置x','10',0.12,0.42)
labreflecty,iptreflecty,labiptreflecty = getlabipt('反射镜位置y','-10',0.12,0.46)
labreflectz,iptreflectz,labiptreflectz = getlabipt('反射镜位置z','0',0.12,0.50)
#压电陶瓷参数设置
labpztnx,iptpztnx,labiptpztnx = getlabipt('陶瓷法向x','0',0.24,0.30)
labpztny,iptpztny,labiptpztny = getlabipt('陶瓷法向y','1',0.24,0.34)
labpztnz,iptpztnz,labiptpztnz = getlabipt('陶瓷法向z','0',0.24,0.38)
labpztx,iptpztx,labiptpztx = getlabipt('陶瓷位置x','0',0.24,0.42)
labpzty,iptpzty,labiptpzty = getlabipt('陶瓷位置y','-20.02003',0.24,0.46)
labpztz,iptpztz,labiptpztz = getlabipt('陶瓷位置z','0',0.24,0.50)
#扩束镜参数设置
labalpha,iptalpha,labiptalpha = getlabipt('扩束镜系数','1',0.0,0.58)

labv1,iptv1,labiptv1 = getlabipt('最小电压','0',0.12,0.58)
labv2,iptv2,labiptv2 = getlabipt('最大电压','10',0.12,0.58)
labvd,iptvd,labiptvd = getlabipt('电压变化步长','1',0.12,0.58)
labv1.place_forget()
labv2.place_forget()
labvd.place_forget()
iptv1.place_forget()
iptv2.place_forget()
iptvd.place_forget()

def _lock():
	iptlamd.place_forget()
	labiptlamd.place(relx=0.18,rely=0.58,anchor='nw')
	labiptlamd.configure(text=iptlamd.get())
	iptN.place_forget()
	labiptN.place(relx=0.30,rely=0.58,anchor='nw')
	labiptN.configure(text=iptN.get())
	iptalpha.place_forget()
	labiptalpha.place(relx=0.06,rely=0.58,anchor='nw')
	labiptalpha.configure(text=iptalpha.get())

	iptscreenx.place_forget()
	labiptscreenx.place(relx=0.06,rely=0.12,anchor='nw')
	labiptscreenx.configure(text=iptscreenx.get())
	iptscreeny.place_forget()
	labiptscreeny.place(relx=0.06,rely=0.16,anchor='nw')
	labiptscreeny.configure(text=iptscreeny.get())
	iptscreenz.place_forget()
	labiptscreenz.place(relx=0.06,rely=0.20,anchor='nw')
	labiptscreenz.configure(text=iptscreenz.get())

	iptscreennx.place_forget()
	labiptscreennx.place(relx=0.06,rely=0.0,anchor='nw')
	labiptscreennx.configure(text=iptscreennx.get())
	iptscreenny.place_forget()
	labiptscreenny.place(relx=0.06,rely=0.04,anchor='nw')
	labiptscreenny.configure(text=iptscreenny.get())
	iptscreennz.place_forget()
	labiptscreennz.place(relx=0.06,rely=0.08,anchor='nw')
	labiptscreennz.configure(text=iptscreennz.get())

	iptlaserx.place_forget()
	labiptlaserx.place(relx=0.18,rely=0.12,anchor='nw')
	labiptlaserx.configure(text=iptlaserx.get())
	iptlasery.place_forget()
	labiptlasery.place(relx=0.18,rely=0.16,anchor='nw')
	labiptlasery.configure(text=iptlasery.get())
	iptlaserz.place_forget()
	labiptlaserz.place(relx=0.18,rely=0.20,anchor='nw')
	labiptlaserz.configure(text=iptlaserz.get())


	iptlaserax.place_forget()
	labiptlaserax.place(relx=0.30,rely=0.12,anchor='nw')
	labiptlaserax.configure(text=iptlaserax.get())
	iptlaseray.place_forget()
	labiptlaseray.place(relx=0.30,rely=0.16,anchor='nw')
	labiptlaseray.configure(text=iptlaseray.get())
	iptlaseraz.place_forget()
	labiptlaseraz.place(relx=0.30,rely=0.20,anchor='nw')
	labiptlaseraz.configure(text=iptlaseraz.get())

	iptlasernx.place_forget()
	labiptlasernx.place(relx=0.18,rely=0.0,anchor='nw')
	labiptlasernx.configure(text=iptlasernx.get())
	iptlaserny.place_forget()
	labiptlaserny.place(relx=0.18,rely=0.04,anchor='nw')
	labiptlaserny.configure(text=iptlaserny.get())
	iptlasernz.place_forget()
	labiptlasernz.place(relx=0.18,rely=0.08,anchor='nw')
	labiptlasernz.configure(text=iptlasernz.get())

	iptlasertx.place_forget()
	labiptlasertx.place(relx=0.30,rely=0.0,anchor='nw')
	labiptlasertx.configure(text=iptlasertx.get())
	iptlaserty.place_forget()
	labiptlaserty.place(relx=0.30,rely=0.04,anchor='nw')
	labiptlaserty.configure(text=iptlaserty.get())
	iptlasertz.place_forget()
	labiptlasertz.place(relx=0.30,rely=0.08,anchor='nw')
	labiptlasertz.configure(text=iptlasertz.get())

	iptsplitx.place_forget()
	labiptsplitx.place(relx=0.06,rely=0.42,anchor='nw')
	labiptsplitx.configure(text=iptsplitx.get())
	iptsplity.place_forget()
	labiptsplity.place(relx=0.06,rely=0.46,anchor='nw')
	labiptsplity.configure(text=iptsplity.get())
	iptsplitz.place_forget()
	labiptsplitz.place(relx=0.06,rely=0.50,anchor='nw')
	labiptsplitz.configure(text=iptsplitz.get())

	iptsplitnx.place_forget()
	labiptsplitnx.place(relx=0.06,rely=0.30,anchor='nw')
	labiptsplitnx.configure(text=iptsplitnx.get())
	iptsplitny.place_forget()
	labiptsplitny.place(relx=0.06,rely=0.34,anchor='nw')
	labiptsplitny.configure(text=iptsplitny.get())
	iptsplitnz.place_forget()
	labiptsplitnz.place(relx=0.06,rely=0.38,anchor='nw')
	labiptsplitnz.configure(text=iptsplitnz.get())

	iptreflectx.place_forget()
	labiptreflectx.place(relx=0.18,rely=0.42,anchor='nw')
	labiptreflectx.configure(text=iptreflectx.get())
	iptreflecty.place_forget()
	labiptreflecty.place(relx=0.18,rely=0.46,anchor='nw')
	labiptreflecty.configure(text=iptreflecty.get())
	iptreflectz.place_forget()
	labiptreflectz.place(relx=0.18,rely=0.50,anchor='nw')
	labiptreflectz.configure(text=iptreflectz.get())

	iptreflectnx.place_forget()
	labiptreflectnx.place(relx=0.18,rely=0.30,anchor='nw')
	labiptreflectnx.configure(text=iptreflectnx.get())
	iptreflectny.place_forget()
	labiptreflectny.place(relx=0.18,rely=0.34,anchor='nw')
	labiptreflectny.configure(text=iptreflectny.get())
	iptreflectnz.place_forget()
	labiptreflectnz.place(relx=0.18,rely=0.38,anchor='nw')
	labiptreflectnz.configure(text=iptreflectnz.get())

	iptpztx.place_forget()
	labiptpztx.place(relx=0.30,rely=0.42,anchor='nw')
	labiptpztx.configure(text=iptpztx.get())
	iptpzty.place_forget()
	labiptpzty.place(relx=0.30,rely=0.46,anchor='nw')
	labiptpzty.configure(text=iptpzty.get())
	iptpztz.place_forget()
	labiptpztz.place(relx=0.30,rely=0.50,anchor='nw')
	labiptpztz.configure(text=iptpztz.get())

	iptpztnx.place_forget()
	labiptpztnx.place(relx=0.30,rely=0.30,anchor='nw')
	labiptpztnx.configure(text=iptpztnx.get())
	iptpztny.place_forget()
	labiptpztny.place(relx=0.30,rely=0.34,anchor='nw')
	labiptpztny.configure(text=iptpztny.get())
	iptpztnz.place_forget()
	labiptpztnz.place(relx=0.30,rely=0.38,anchor='nw')
	labiptpztnz.configure(text=iptpztnz.get())

	labv1.place(relx=0.0,rely=0.8,anchor='nw')
	labv2.place(relx=0.12,rely=0.8,anchor='nw')
	labvd.place(relx=0.24,rely=0.8,anchor='nw')
	iptv1.place(relx=0.06,rely=0.8,anchor='nw',width=70,height=20)
	iptv2.place(relx=0.18,rely=0.8,anchor='nw')
	iptvd.place(relx=0.30,rely=0.8,anchor='nw',width=70,height=20)
	labV.place(relx=0.0,rely=0.75,anchor='nw')
	iptV.place(relx=0.06,rely=0.75,anchor='nw',width=70,height=20)
	btnlock.place_forget()
	btnunlock.place(relx=0.18,rely=0.68,anchor='nw')

	btnplot.place_forget()
	btnplot2.place(relx=0.12,rely=0.75,anchor='nw')
	btnplot3.place(relx=0.36,rely=0.8,anchor='nw')

	lamd=float(iptlamd.get())
	N=float(iptN.get())
	hlaser=0.2
	wlaser=0.2
	llaser=0.2
	laserx=float(iptlaserx.get())
	lasery=float(iptlasery.get())
	laserz=float(iptlaserz.get())
	lasernx=float(iptlasernx.get())
	laserny=float(iptlaserny.get())
	lasernz=float(iptlasernz.get())
	lasertx=float(iptlasertx.get())
	laserty=float(iptlaserty.get())
	lasertz=float(iptlasertz.get())
	laserax=float(iptlaserax.get())
	laseray=float(iptlaseray.get())
	laseraz=float(iptlaseraz.get())
	splitx=float(iptsplitx.get())
	splity=float(iptsplity.get())
	splitz=float(iptsplitz.get())
	splitnx=float(iptsplitnx.get())
	splitny=float(iptsplitny.get())
	splitnz=float(iptsplitnz.get())
	splitn2=splitx*splitnx+splity*splitny+splitz*splitnz
	reflectx=float(iptreflectx.get())
	reflecty=float(iptreflecty.get())
	reflectz=float(iptreflectz.get())
	reflectnx=float(iptreflectnx.get())
	reflectny=float(iptreflectny.get())
	reflectnz=float(iptreflectnz.get())
	reflectn2=reflectx*reflectnx+reflecty*reflectny+reflectz*reflectnz
	pztx=float(iptpztx.get())
	pzty=float(iptpzty.get())
	pztz=float(iptpztz.get())
	pztnx=float(iptpztnx.get())
	pztny=float(iptpztny.get())
	pztnz=float(iptpztnz.get())
	Dp=sympy.Symbol('dp')
	pztx+=pztnx*Dp
	pzty+=pztny*Dp
	pztz+=pztnz*Dp
	pztn2=pztx*pztnx+pzty*pztny+pztz*pztnz
	screenx=float(iptscreenx.get())
	screeny=float(iptscreeny.get())
	screenz=float(iptscreenz.get())
	screennx=float(iptscreennx.get())
	screenny=float(iptscreenny.get())
	screennz=float(iptscreennz.get())
	screenn2=screenx*screennx+screeny*screenny+screenz*screennz
	alpha=float(iptalpha.get())
	wb=int(N**0.5)
	a.clear()
	rangy=5
	rangz=5
	wb=int(N**0.5)

	x0 = sympy.Symbol('x')
	y0 = sympy.Symbol('y')
	z0 = sympy.Symbol('z')
	nx0= sympy.Symbol('nx')
	ny0= sympy.Symbol('ny')
	nz0= sympy.Symbol('nz')
	
	x1n=x0
	y1n=y0
	z1n=z0
	nx1n=nx0
	ny1n=ny0
	nz1n=nz0
	step1n=0

	x2n=x0
	y2n=y0
	z2n=z0
	nx2n=nx0
	ny2n=ny0
	nz2n=nz0
	step2n=0

	x1n,y1n,z1n,nx1n,ny1n,nz1n,step1n,x2n,y2n,z2n,nx2n,ny2n,nz2n,step2n=LaserToSplit(x1n,y1n,z1n,nx1n,ny1n,nz1n,step1n,x2n,y2n,z2n,nx2n,ny2n,nz2n,step2n,splitn2,splitnx,splitny,splitnz)
	#print('laser to split')

	x1n,y1n,z1n,nx1n,ny1n,nz1n,step1n=betweeninv(x1n,y1n,z1n,nx1n,ny1n,nz1n,step1n,pztn2,pztnx,pztny,pztnz)
	#print('split to pzt')

	x2n,y2n,z2n,nx2n,ny2n,nz2n,step2n=betweeninv(x2n,y2n,z2n,nx2n,ny2n,nz2n,step2n,reflectn2,reflectnx,reflectny,reflectnz)
	#print('split to reflect')

	x1n,y1n,z1n,nx1n,ny1n,nz1n,step1n=betweeninv(x1n,y1n,z1n,nx1n,ny1n,nz1n,step1n,splitn2,splitnx,splitny,splitnz)
	#print('pzt to split')

	x1n,y1n,z1n,nx1n,ny1n,nz1n,step1n=betweeninv(x1n,y1n,z1n,nx1n,ny1n,nz1n,step1n,screenn2,screennx,screenny,screennz)
	#print('pzt-split to screen')

	x2n,y2n,z2n,nx2n,ny2n,nz2n,step2n=betweeninv(x2n,y2n,z2n,nx2n,ny2n,nz2n,step2n,screenn2,screennx,screenny,screennz)
	#print('reflect to screen')
	#print('light inited')
	global expry1n
	global exprz1n
	global exprstep1n
	global expry2n
	global exprz2n
	global exprstep2n
	global locked

	global Galpha
	global Gwb
	global Gllaser
	global Ghlaser
	global Glaserx
	global Glasery
	global Glaserz
	global Glasernx
	global Glaserny
	global Glasernz
	global Glaserax
	global Glaseray
	global Glaseraz
	global Glasertx
	global Glaserty
	global Glasertz
	global Grangy
	global Grangz
	global Glamd
	global Gscreenx
	global Gscreeny
	global Gscreenz
	locked=True
	expry1n=y1n
	exprz1n=z1n
	exprstep1n=step1n
	expry2n=y2n
	exprz2n=z2n
	exprstep2n=step2n
	Galpha=alpha
	Gwb=wb
	Gllaser=llaser
	Ghlaser=hlaser
	Glaserx=laserx
	Glasery=lasery
	Glaserz=laserz
	Glasernx=lasernx
	Glaserny=laserny
	Glasernz=lasernz
	Glaserax=laserax
	Glaseray=laseray
	Glaseraz=laseraz
	Glasertx=lasertx
	Glaserty=laserty
	Glasertz=lasertz
	Grangy=rangy
	Grangz=rangz
	Glamd=lamd
	Gscreenx=screenx
	Gscreeny=screeny
	Gscreenz=screenz

def _unlock():
	global locked
	locked=False
	labV.place_forget()
	iptV.place_forget()
	labiptlamd.place_forget()
	iptlamd.place(relx=0.18,rely=0.58,anchor='nw',width=70,height=20)

	labiptN.place_forget()
	iptN.place(relx=0.30,rely=0.58,anchor='nw',width=70,height=20)

	labiptalpha.place_forget()
	iptalpha.place(relx=0.06,rely=0.58,anchor='nw',width=70,height=20)


	labiptscreenx.place_forget()
	iptscreenx.place(relx=0.06,rely=0.12,anchor='nw',width=70,height=20)

	labiptscreeny.place_forget()
	iptscreeny.place(relx=0.06,rely=0.16,anchor='nw',width=70,height=20)

	labiptscreenz.place_forget()
	iptscreenz.place(relx=0.06,rely=0.20,anchor='nw',width=70,height=20)


	labiptscreennx.place_forget()
	iptscreennx.place(relx=0.06,rely=0.0,anchor='nw',width=70,height=20)

	labiptscreenny.place_forget()
	iptscreenny.place(relx=0.06,rely=0.04,anchor='nw',width=70,height=20)

	labiptscreennz.place_forget()
	iptscreennz.place(relx=0.06,rely=0.08,anchor='nw',width=70,height=20)


	labiptlaserx.place_forget()
	iptlaserx.place(relx=0.18,rely=0.12,anchor='nw',width=70,height=20)

	labiptlasery.place_forget()
	iptlasery.place(relx=0.18,rely=0.16,anchor='nw',width=70,height=20)

	labiptlaserz.place_forget()
	iptlaserz.place(relx=0.18,rely=0.20,anchor='nw',width=70,height=20)



	labiptlaserax.place_forget()
	iptlaserax.place(relx=0.30,rely=0.12,anchor='nw',width=70,height=20)

	labiptlaseray.place_forget()
	iptlaseray.place(relx=0.30,rely=0.16,anchor='nw',width=70,height=20)

	labiptlaseraz.place_forget()
	iptlaseraz.place(relx=0.30,rely=0.20,anchor='nw',width=70,height=20)


	labiptlasernx.place_forget()
	iptlasernx.place(relx=0.18,rely=0.0,anchor='nw',width=70,height=20)

	labiptlaserny.place_forget()
	iptlaserny.place(relx=0.18,rely=0.04,anchor='nw',width=70,height=20)

	labiptlasernz.place_forget()
	iptlasernz.place(relx=0.18,rely=0.08,anchor='nw',width=70,height=20)


	labiptlasertx.place_forget()
	iptlasertx.place(relx=0.30,rely=0.0,anchor='nw',width=70,height=20)

	labiptlaserty.place_forget()
	iptlaserty.place(relx=0.30,rely=0.04,anchor='nw',width=70,height=20)

	labiptlasertz.place_forget()
	iptlasertz.place(relx=0.30,rely=0.08,anchor='nw',width=70,height=20)


	labiptsplitx.place_forget()
	iptsplitx.place(relx=0.06,rely=0.42,anchor='nw',width=70,height=20)

	labiptsplity.place_forget()
	iptsplity.place(relx=0.06,rely=0.46,anchor='nw',width=70,height=20)

	labiptsplitz.place_forget()
	iptsplitz.place(relx=0.06,rely=0.50,anchor='nw',width=70,height=20)


	labiptsplitnx.place_forget()
	iptsplitnx.place(relx=0.06,rely=0.30,anchor='nw',width=70,height=20)

	labiptsplitny.place_forget()
	iptsplitny.place(relx=0.06,rely=0.34,anchor='nw',width=70,height=20)

	labiptsplitnz.place_forget()
	iptsplitnz.place(relx=0.06,rely=0.38,anchor='nw',width=70,height=20)


	labiptreflectx.place_forget()
	iptreflectx.place(relx=0.18,rely=0.42,anchor='nw',width=70,height=20)

	labiptreflecty.place_forget()
	iptreflecty.place(relx=0.18,rely=0.46,anchor='nw',width=70,height=20)

	labiptreflectz.place_forget()
	iptreflectz.place(relx=0.18,rely=0.50,anchor='nw',width=70,height=20)


	labiptreflectnx.place_forget()
	iptreflectnx.place(relx=0.18,rely=0.30,anchor='nw',width=70,height=20)

	labiptreflectny.place_forget()
	iptreflectny.place(relx=0.18,rely=0.34,anchor='nw',width=70,height=20)

	labiptreflectnz.place_forget()
	iptreflectnz.place(relx=0.18,rely=0.38,anchor='nw',width=70,height=20)


	labiptpztx.place_forget()
	iptpztx.place(relx=0.30,rely=0.42,anchor='nw',width=70,height=20)

	labiptpzty.place_forget()
	iptpzty.place(relx=0.30,rely=0.46,anchor='nw',width=70,height=20)

	labiptpztz.place_forget()
	iptpztz.place(relx=0.30,rely=0.50,anchor='nw',width=70,height=20)


	labiptpztnx.place_forget()
	iptpztnx.place(relx=0.30,rely=0.30,anchor='nw',width=70,height=20)

	labiptpztny.place_forget()
	iptpztny.place(relx=0.30,rely=0.34,anchor='nw',width=70,height=20)

	labiptpztnz.place_forget()
	iptpztnz.place(relx=0.30,rely=0.38,anchor='nw',width=70,height=20)
	labv1.place_forget()
	labv2.place_forget()
	labvd.place_forget()
	iptv1.place_forget()
	iptv2.place_forget()
	iptvd.place_forget()

	btnunlock.place_forget()
	btnlock.place(relx=0.18,rely=0.68,anchor='nw')

	btnplot2.place_forget()
	btnplot3.place_forget()
	btnplot.place(relx=0.0,rely=0.68,anchor='nw')

btnlock = tkinter.Button(master=root, text="锁定参数", command=_lock)
btnlock.place(relx=0.18, rely=0.68, anchor='nw')

btnunlock = tkinter.Button(master=root, text="解锁锁定", command=_unlock)
btnunlock.place_forget()


root.mainloop()