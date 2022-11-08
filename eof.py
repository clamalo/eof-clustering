import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import xarray as xr

from eofs.xarray import Eof
from ecmwf.opendata import Client


working_directory = '/Users/clamalo/documents/clustering/'



def create_colormap():
    cmap = colors.ListedColormap(['#000046', '#243877', '#446aa1', '#6498c3', '#84b9d8','#a4cfe4','#c2e3ef','#ffffff','#fec376','#fca35f','#f7804c','#ea5d3b','#c73d29','#941d18','#540000'])
    return cmap

def ingest_ecmwf(frame):
    client = Client("ecmwf", beta=True)

    parameters = ['gh']
    filename = working_directory+'gribs/500_f'+str(frame)+'.grib'

    client.retrieve(
        date=-1,
        time=0,
        step=int(frame),
        stream="enfo",
        type="pf",
        levtype="pl",
        levelist=[500],
        param=parameters,
        target=filename
    )

def eofs_from_dataset(sst):
    coslat = np.cos(np.deg2rad(sst.coords['latitude'].values))
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(sst, weights=wgts)

    eofs = solver.eofsAsCorrelation(neofs=2)
    pcs = solver.pcs(npcs=2, pcscaling=1)

    return eofs, pcs

def plot_eofs(eofs):
    clevs = np.linspace(-1, 1, 11)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=190))
    fill = eofs[0].plot.contourf(ax=ax, levels=clevs, cmap=plt.cm.RdBu_r,
                                add_colorbar=False, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='w', edgecolor='k')
    cb = plt.colorbar(fill, orientation='horizontal')
    cb.set_label('Correlation Coefficient', fontsize=12)
    ax.coastlines()
    ax.set_title('EOF1 expressed as correlation', fontsize=16)
    plt.savefig('/Users/clamalo/documents/clustering/images/eof1.png',dpi=300,bbox_inches='tight')
    plt.clf()

    clevs = np.linspace(-1, 1, 11)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=190))
    fill = eofs[1].plot.contourf(ax=ax, levels=clevs, cmap=plt.cm.RdBu_r,
                                add_colorbar=False, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='w', edgecolor='k')
    cb = plt.colorbar(fill, orientation='horizontal')
    cb.set_label('Correlation Coefficient', fontsize=12)
    ax.coastlines()
    ax.set_title('EOF2 expressed as correlation', fontsize=16)
    plt.savefig('/Users/clamalo/documents/clustering/images/eof2.png',dpi=300,bbox_inches='tight')
    plt.clf()

def plot_pcs_scatter(pcs):
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(pcs[:, 0], pcs[:, 1], c=sst.time.values, cmap='jet')
    plt.ylim(-3, 3)
    plt.xlim(-3, 3)
    plt.savefig('/Users/clamalo/documents/clustering/images/pcs.png',dpi=300,bbox_inches='tight')
    plt.clf()


def cluster(pcs):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=0).fit(pcs)

    # fig = plt.figure(figsize=(5, 5))
    # plt.scatter(pcs[:, 0], pcs[:, 1], c=kmeans.labels_, cmap='jet')
    # plt.ylim(-3, 3)
    # plt.xlim(-3, 3)
    # plt.savefig('/Users/clamalo/documents/clustering/images/kmeans_cluster.png',dpi=300,bbox_inches='tight')

    return kmeans.labels_, kmeans.cluster_centers_


for frame in range(0,145,6):
    ingest_ecmwf(frame)

for frame in range(0,121,24):

    filenames = ['/Users/clamalo/documents/clustering/gribs/500_f'+str(frame)+'.grib','/Users/clamalo/documents/clustering/gribs/500_f'+str(frame+6)+'.grib','/Users/clamalo/documents/clustering/gribs/500_f'+str(frame+12)+'.grib','/Users/clamalo/documents/clustering/gribs/500_f'+str(frame+18)+'.grib']
    datasets = []
    for filename in filenames:
        ds = xr.open_dataset(filename, engine='cfgrib')
        datasets.append(ds)

    ds = xr.concat(datasets, dim='time')
    ds = ds.mean(dim='time')

    ds = ds.sel(latitude=slice(60,30),longitude=slice(-140,-90))
    mean_ds = ds.mean(dim='number')
    ds['anomaly_from_mean'] = ds['gh'] - mean_ds['gh']
    sst = ds['gh']
    sst = sst.rename({'number':'time'})

    eofs,pcs = eofs_from_dataset(sst)

    # plot_eofs(eofs)
    # plot_pcs_scatter(pcs)

    labels,centers = cluster(pcs)

    fig = plt.figure(figsize=(5, 5))
    plt.scatter(pcs[:, 0], pcs[:, 1], c=labels, cmap='jet')
    plt.ylim(-3, 3)
    plt.xlim(-3, 3)
    plt.savefig('/Users/clamalo/documents/clustering/images/pcs_kmeans.png',dpi=300,bbox_inches='tight')
    plt.clf()


    cluster_1_datasets = []
    cluster_2_datasets = []
    cluster_3_datasets = []
    cluster_4_datasets = []

    for i in range(len(labels)):

        i_ds = ds.isel(number=i)

        if labels[i] == 0:
            cluster_1_datasets.append(i_ds)
        elif labels[i] == 1:
            cluster_2_datasets.append(i_ds)
        elif labels[i] == 2:
            cluster_3_datasets.append(i_ds)
        elif labels[i] == 3:
            cluster_4_datasets.append(i_ds)

    cluster_1_len = len(cluster_1_datasets)
    cluster_2_len = len(cluster_2_datasets)
    cluster_3_len = len(cluster_3_datasets)
    cluster_4_len = len(cluster_4_datasets)
    lens = [cluster_1_len,cluster_2_len,cluster_3_len,cluster_4_len]
    lens = sorted(lens,reverse=True)

    #take the mean of each cluster
    cluster_1_mean = xr.concat(cluster_1_datasets, dim='number')
    cluster_2_mean = xr.concat(cluster_2_datasets, dim='number')
    cluster_3_mean = xr.concat(cluster_3_datasets, dim='number')
    cluster_4_mean = xr.concat(cluster_4_datasets, dim='number')

    #organize cluster means into list sorted by cluster_x_len
    cluster_means = [cluster_1_mean,cluster_2_mean,cluster_3_mean,cluster_4_mean]
    cluster_means.sort(key=lambda x: x['number'].size, reverse=True)
    for c in range(len(cluster_means)):
        cluster_means[c] = cluster_means[c].mean(dim='number')

    colormap = create_colormap()
    bounds = [-250,-200,-150,-100,-75,-50,-25,25,50,75,100,150,200,250]
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=15, extend='both')

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=190)})
    fig.suptitle('Cluster Differences From Ensemble Mean - Hours '+str(frame)+'-'+str(frame+24), fontsize=16)

    variable_to_plot = 'anomaly_from_mean'

    axes[0, 0].set_title('Cluster 1: '+str(lens[0])+' ('+str(int((lens[0]/50)*100))+'%)', fontsize=16)
    axes[0, 0].contourf(cluster_means[0][variable_to_plot].longitude, cluster_means[0][variable_to_plot].latitude, cluster_means[0][variable_to_plot], transform=ccrs.PlateCarree(), cmap=colormap, norm=norm, levels=bounds)
    axes[0, 0].contour(cluster_means[0]['gh'].longitude, cluster_means[0]['gh'].latitude, cluster_means[0]['gh'], transform=ccrs.PlateCarree(),colors='k')
    axes[0, 0].coastlines()
    axes[0, 0].add_feature(cfeature.STATES, edgecolor='k', linewidth=0.5)

    axes[0, 1].set_title('Cluster 2: '+str(lens[1])+' ('+str(int((lens[1]/50)*100))+'%)', fontsize=16)
    axes[0, 1].contourf(cluster_means[1][variable_to_plot].longitude, cluster_means[1][variable_to_plot].latitude, cluster_means[1][variable_to_plot], transform=ccrs.PlateCarree(), cmap=colormap, norm=norm, levels=bounds)
    axes[0, 1].contour(cluster_means[1]['gh'].longitude, cluster_means[1]['gh'].latitude, cluster_means[1]['gh'], transform=ccrs.PlateCarree(),colors='k')
    axes[0, 1].coastlines()
    axes[0, 1].add_feature(cfeature.STATES, edgecolor='k', linewidth=0.5)

    axes[1, 0].set_title('Cluster 3: '+str(lens[2])+' ('+str(int((lens[2]/50)*100))+'%)', fontsize=16)
    axes[1, 0].contourf(cluster_means[2][variable_to_plot].longitude, cluster_means[2][variable_to_plot].latitude, cluster_means[2][variable_to_plot], transform=ccrs.PlateCarree(), cmap=colormap, norm=norm, levels=bounds)
    axes[1, 0].contour(cluster_means[2]['gh'].longitude, cluster_means[2]['gh'].latitude, cluster_means[2]['gh'], transform=ccrs.PlateCarree(),colors='k')
    axes[1, 0].coastlines()
    axes[1, 0].add_feature(cfeature.STATES, edgecolor='k', linewidth=0.5)

    axes[1, 1].set_title('Cluster 4: '+str(lens[3])+' ('+str(int((lens[3]/50)*100))+'%)', fontsize=16)
    axes[1, 1].contourf(cluster_means[3][variable_to_plot].longitude, cluster_means[3][variable_to_plot].latitude, cluster_means[3][variable_to_plot], transform=ccrs.PlateCarree(), cmap=colormap, norm=norm, levels=bounds)
    axes[1, 1].contour(cluster_means[3]['gh'].longitude, cluster_means[3]['gh'].latitude, cluster_means[3]['gh'], transform=ccrs.PlateCarree(),colors='k')
    axes[1, 1].coastlines()
    axes[1, 1].add_feature(cfeature.STATES, edgecolor='k', linewidth=0.5)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=cbar_ax, ticks=bounds)
    cb.ax.tick_params(labelsize=14)
    cb.set_label('500mb Height Departure From Ensemble Mean (m)', fontsize=16)

    plt.savefig('/Users/clamalo/documents/clustering/images/clusters_f'+str(frame)+'.png',dpi=300,bbox_inches = 'tight')
    plt.clf()