from readimc import MCDFile, TXTFile

def read_imc_dict(file_big, id):
    #return a dictionary with three images: slide, panorama, imc
    with MCDFile(file_big) as f:
        slide = f.slides[0] # first slide
        panorama = slide.panoramas[0] # first panorama of first slide
        acquisition = slide.acquisitions[id]
        before_ablation = f.read_before_ablation_image(acquisition)
        
        # print panorama info
        # print("Panorama info:")
        # print(panorama.width_um, panorama.height_um)
        
        # print acquisition info
        print("Acquisition info:")
        print(acquisition.id, acquisition.description, acquisition.width_um, acquisition.height_um)
        
        channel_labels = acquisition.channel_labels
        
        img_slide = f.read_slide(slide)
        img_panorama = f.read_panorama(panorama)
        img_imc = f.read_acquisition(acquisition)

        # put the image into a dictionary
        img_dict = {}
        img_dict['slide'] = img_slide
        img_dict['panorama'] = img_panorama
        img_dict['description'] = acquisition.description
        img_dict['imc'] = img_imc
        img_dict['bf_imc'] = before_ablation
    return img_dict, channel_labels