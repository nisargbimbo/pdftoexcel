import streamlit as st
from paddleocr import PaddleOCR,draw_ocr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdf2image import convert_from_path, convert_from_bytes
import os
import warnings
warnings.filterwarnings("ignore")
from stqdm import stqdm
import glob
import tempfile
import pathlib

# Load the OCR Model
ocr = PaddleOCR(use_angle_cls=False, lang='en', det_model_dir="./PaddleOCR/en_PP-OCRv3_det_infer",
                rec_model_dir="./PaddleOCR/en_PP-OCRv3_rec_infer",
                cls_model_dir ="./PaddleOCR/ch_ppocr_mobile_v2.0_cls_infer",
                show_log = False,
                e2e_pgnet_score_thresh=0.4) # need to run only once to download and load model into memory

temp_folder = tempfile.TemporaryDirectory()
save_folder = pathlib.Path(temp_folder.name)

def convert_pdf_image_path(pdf_path, save_folder, filename, number_of_pages):
    
    # Store Pdf with convert_from_path function
    images = convert_from_path(pdf_path) #, poppler_path = './poppler-23.07.0/Library/bin')
    
    # if os.path.isdir(os.path.join(save_folder, filename[:-4])):
    #     pass
    # else:
    #     os.mkdir(os.path.join(save_folder, filename[:-4]))

    if number_of_pages != 0:
        for i in range(int(number_of_pages)):
            # Save pages as images in the pdf
            images[i].save(os.path.join(save_folder, 'page_'+ str(i) +'.png'), 'PNG')
        
    else:
        for i in range(len(images)):
            # Save pages as images in the pdf
            images[i].save(os.path.join(save_folder, 'page_'+ str(i) +'.png'), 'PNG')



def convert_pdf_image_bytes(pdf_file, save_folder, filename, number_of_pages):
    
    # Store Pdf with convert_from_path function
    images = convert_from_bytes(pdf_file.read()) #, poppler_path = './poppler-23.07.0/Library/bin')

    # uploaded_file_path = pathlib.Path(temp_dir.name) / filename

    #     output_temporary_file.write(uploaded_file.read())
    
    # if os.path.isdir(os.path.join(save_folder, filename[:-4])):
    #     pass
    # else:
    #     os.mkdir(os.path.join(save_folder, filename[:-4]))
    
    if number_of_pages != 0:
        for i in range(int(number_of_pages)):
                # Save pages as images in the pdf
                images[i].save(os.path.join(save_folder, 'page_'+ str(i) +'.png'), 'PNG')
    
    else:
        for i in range(len(images)):
            # Save pages as images in the pdf
            images[i].save(os.path.join(save_folder, 'page_'+ str(i) +'.png'), 'PNG')


def convert_pdf_image_to_excel(save_folder, filename):

    ctr = 1
    final_df = pd.DataFrame()
    filelist = glob.glob(os.path.join(save_folder, "*.png"))
    
    #for images in stqdm(glob.glob(os.path.join(save_folder, filename[:-4]))):
    for i in range(len(filelist)):

        images = "page_{}.png".format(i)

        if images.endswith('.jpg') or images.endswith('.png'):
            if images[:-4] == 'page_0':
                continue
            
            img_path = os.path.join(save_folder, images)
            result = ocr.ocr(img_path, cls=False)
            
            for i in range(len(result[0])):
                if result[0][i][1][0].startswith('Time') or result[0][i][1][0].startswith('Date'):
                    print("Yes")
                    skip_page = 1
                    break
                else:
                    skip_page = 0
                
            if skip_page == 0:

                result_df = pd.DataFrame(columns = ['Text', 
                                            'BB_1_x', 'BB_1_y', 
                                            'BB_2_x', 'BB_2_y', 
                                            'BB_3_x', 'BB_3_y', 
                                            'BB_4_x', 'BB_4_y'])
                
                for i in range(len(result[0])):

                    new_row = pd.DataFrame({'Text' : result[0][i][1][0], 
                                            'BB_1_x': result[0][i][0][0][0],
                                            'BB_1_y': result[0][i][0][0][1], 
                                            'BB_2_x': result[0][i][0][1][0], 
                                            'BB_2_y': result[0][i][0][1][1], 
                                            'BB_3_x': result[0][i][0][2][0], 
                                            'BB_3_y': result[0][i][0][2][1], 
                                            'BB_4_x': result[0][i][0][3][0], 
                                            'BB_4_y': result[0][i][0][3][1]
                                            }, index=[i])
                    
                    result_df = pd.concat([result_df, new_row], ignore_index = True)

                # Define bin edges
                bins_y = [400, 465, 530, 595, 660, 725, 790, 855, 920, 985, 1050, 1115, 1180, 1245, 1310, 1375, 1440, 1505,
                        1570, 1635, 1700, 1765, 1830, 1895, 1960, 2025]
                
                bins_x = [0, 200, 330, 720, 875, 1040]

                bin_labels_y = ['BBox0',  'BBox1', 'BBox2', 'BBox3', 'BBox4', 'BBox5', 'BBox6', 'BBox7', 'BBox8', 'BBox9', 'BBox10',
                            'BBox11', 'BBox12', 'BBox13', 'BBox14', 'BBox15', 'BBox16', 'BBox17', 'BBox18', 'BBox19', 'BBox20', 
                            'BBox21', 'BBox22', 'BBox23', 'BBox24']
                
                bin_labels_x = ['Column0', 'Column1', 'Column2', 'Column3', 'Column4']

                # Assign values to bins with custom labels
                result_df['bins_y'] = pd.cut(result_df['BB_1_y'], bins=bins_y, labels=bin_labels_y)
                result_df['bins_x'] = pd.cut(result_df['BB_1_x'], bins=bins_x, labels=bin_labels_x)

                result_df.dropna(subset=['bins_y'], inplace = True)
                grouped_result = result_df.groupby(by = result_df['bins_y'])
                grouped_result = grouped_result.filter(lambda group: len(group) >= 4)
                grouped_result = grouped_result.sort_values(by = ['bins_y', 'bins_x'])
                grouped_result = grouped_result.groupby(by = grouped_result['bins_y'])
                combined_texts = grouped_result['Text'].apply(lambda group: group.tolist())
                combined_df = pd.DataFrame(combined_texts)
                df_split = combined_df['Text'].apply(pd.Series)
                df_split.rename(columns={0: 'Article', 1: 'Entr.', 2: 'Description', 3: 'Qty in Hand', 4: 'Qty Sold'}, inplace=True)
                df_split.dropna(inplace =True, how = 'all')
                df_split.reset_index(inplace=True, drop=True)
                df_split['Article'] = df_split['Article'].str.replace('[\.\s]', '', regex=True)
                df_split['Entr.'] = df_split['Entr.'].str.replace('[\.\s]', '', regex=True)
                df_split['Qty in Hand'] = df_split['Qty in Hand'].str.replace('[\.\s]', '', regex=True)
                df_split['Qty Sold'] = df_split['Qty Sold'].str.replace('[\.\s]', '', regex=True)
                df_split['Qty in Hand'] = df_split['Qty in Hand'].apply(lambda x: '-' + x[:-1] if str(x).endswith('-') else x)
                df_split['Qty in Hand'].fillna(1, inplace = True)
                df_split['Qty in Hand'].replace(':', 8, inplace = True)
                df_split['Qty in Hand'].replace('-', -8, inplace = True)
                df_split['Qty Sold'].fillna(1, inplace = True)
                df_split['Qty Sold'].replace(':', 8, inplace = True)
                df_split['Qty Sold'].replace('-', -8, inplace = True)
                df_split['Qty in Hand'] = pd.to_numeric(df_split['Qty in Hand'], errors = 'coerce')
                df_split['Qty Sold'] = pd.to_numeric(df_split['Qty Sold'], errors = 'coerce')
                df_split['Entr.'] = pd.to_numeric(df_split['Entr.'], errors = 'coerce')
                
                df_split.sort_values(by = ['Article', 'Entr.'], inplace = True)

                final_df = pd.concat([final_df, df_split])
            
        st.write("Page {} converted successfully!!".format(ctr))
        ctr += 1

    final_df.reset_index(inplace=True, drop=True)
    final_df.dropna(inplace=True, how = 'all')

    subtotals = final_df.groupby('Article')['Qty Sold'].sum()
    subtotal_rows = pd.DataFrame()
    for article, subtotal in subtotals.items():
        subtotal = subtotals.loc[article]
        new_row  = pd.DataFrame({'Article': [article], 'Entr.': [''], 'Description': ['Total Article:'], 'Qty in Hand' : [''], 'Qty Sold': [subtotal]})
        subtotal_rows = pd.concat([subtotal_rows, new_row], ignore_index=True)

    # final_df['num Qty in Hand'] = pd.to_numeric(final_df['Qty in Hand'], errors = 'coerce')
    # final_df['num Qty Sold'] = pd.to_numeric(final_df['Qty Sold'], errors = 'coerce')
    # non_num_inhand_index = final_df[final_df['num Qty in Hand'].isna()].index
    # non_num_sold_index = final_df[final_df['num Qty Sold'].isna()].index
    #final_df.sort_values(by = ['Article', 'Entr.'])

    final_df = pd.concat([final_df, subtotal_rows])
    final_df['Article'] = pd.to_numeric(final_df['Article'], errors = 'coerce')

    final_df.sort_values(by = ['Article', 'Entr.'], ascending=[True, True], inplace = True)
    final_df.reset_index(inplace=True, drop=True)

    download1 = st.download_button(
    label="Download converted file as CSV",
    data=final_df.to_csv(index=False).encode('utf-8'),
    file_name='{}.csv'.format(filename[:-4]),
    mime='text/csv'
    )

    # if os.path.isdir(os.path.join(save_folder, "convertedToExcel")):
    #     try:
    #         final_df.to_csv(os.path.join(save_folder, "convertedToExcel", '{}.csv'.format(filename[:-4])))
    #         st.write("### {} file conversion is completed and output is saved".format(filename[:-4])) 

    #     except Exception as error:
    #         if type(error).__name__ == 'PermissionError':
    #             st.write("#### Error!! Close the excel file which has the same as the pdf and try again.")

    # else:
    #     os.mkdir(os.path.join(save_folder, "convertedToExcel"))
    #     try:
    #         final_df.to_csv(os.path.join(save_folder, "convertedToExcel", '{}.csv'.format(filename[:-4])))
    #         st.write("### {} file conversion is completed and output is saved".format(filename[:-4])) 

    #     except Exception as error:
    #         if type(error).__name__ == 'PermissionError':
    #             st.write("#### Error!! Close the excel file which has the same as the pdf aad try again.")



# Streamlit Compoenets
# Set page config
apptitle = 'FAX Reading'
st.set_page_config(page_title=apptitle, page_icon=":eyeglasses:")

# Title the app
st.title('XLSify: FAX PDF to Excel Converter')

st.markdown("""
##### This application will help you convert the FAX received from Customers into Excel directly.

##### Currently implemented for Costco FAX.
""")

st.sidebar.markdown("## Select the option from below")

#-- Set time by GPS or event
select_event = st.sidebar.selectbox('How do you want to convert the PDF?',
                                    ['Uploading the PDF File']) #, 'Convert PDF/s file from folder'])

select_what_to_do = st.sidebar.selectbox('How many pages you want to convert',
                                    ['Enter number of pages', 'All Pages']) #, 'Convert PDF/s file from folder'])



if select_what_to_do == 'Enter number of pages':

    #save_folder = st.sidebar.text_input('### Enter the save folder path', key = '3')
    number_of_pages = st.sidebar.text_input('### Enter the number of pages you want to convert', key = 'number_of_pages')


if select_what_to_do == 'All Pages':

    number_of_pages = 0

print(number_of_pages)

if select_event == 'Uploading the PDF File':

    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

    if st.sidebar.button('Convert the PDF'):

        if uploaded_file is None: #or len(save_folder) == 0:
            st.write("### Please upload the pdf or specify the save folder path...")
        
        else:
            filename = uploaded_file.name
                
            #else:
            convert_pdf_image_bytes(uploaded_file, save_folder, filename, number_of_pages)
            st.write("PDF converted to Images Sucessfully!!")
            convert_pdf_image_to_excel(save_folder, filename)
    
if select_event == 'Convert PDF/s file from folder':

    pdf_folder = st.text_input('### Enter the pdf folder path', key = '4')

    if st.sidebar.button('Convert the PDF'):

        if len(pdf_folder) == 0 or len(save_folder) == 0:
            st.write("### Specify the PDF folder path or specify the save folder path...")
        
        else:
            for filename in (os.listdir(pdf_folder)):
                    
                pdf_path = os.path.join(pdf_folder, filename)
                convert_pdf_image_path(pdf_path, save_folder, filename, number_of_pages)
                st.write("PDF converted to Images Sucessfully!!")
                convert_pdf_image_to_excel(save_folder, filename)