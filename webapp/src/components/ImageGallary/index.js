import React, { useState } from "react";
import { Box, Typography } from "@material-ui/core";
import Gallery from "react-grid-gallery";
import useStyles from "screens/FeedBackLoop/styles";

const ImageGallary = (props) => {
    const classes = useStyles();
    const { keyframe_labels } = props?.selectedRowData?.item;
    let arrayImg = [],
        xlsxImg = [];
    keyframe_labels[0]?.forEach((item, Idx) => {
        let temp = {
            thumbnailWidth: 320,
            thumbnailHeight: 172,
            width: 320,
            height: 172,
            src: "",
            caption: "",
            confidenceScore: "",
        };
        temp["src"] = props.imageUrl;
        temp["thumbnail"] = props.imageUrl;
        if (item.split(".")[1] === "xlsx") {
            xlsxImg.push(temp);
        } else {
            arrayImg.push(temp);
        }
    });
    keyframe_labels[1]?.forEach((item, Idx) => {
        arrayImg[Idx]["caption"] = item;
    });
    keyframe_labels[2]?.forEach((item, idxTemp) => {
        arrayImg[idxTemp]["confidenceScore"] = item;
    });
    const images = [arrayImg, xlsxImg];
    const [currentImage, setCurrentImage] = useState(0);
    const onCurrentImageChange = (index) => {
        setCurrentImage(index);
    };

    const ConentData = (props) => {
        const { imageArr } = props;
        return (
            <Box
                component="div"
                display="block"
                style={{
                    display: "block",
                    minHeight: "1",
                    width: "100%",
                    overflow: "auto",
                    margin: "10px",
                }}
            >
                <Gallery
                    images={imageArr}
                    enableLightbox={true}
                    enableImageSelection={true}
                    currentImageWillChange={onCurrentImageChange}
                    showLightboxThumbnails={true}
                    lightboxWidth={1536}
                    // customControls={[
                    //   <button key="deleteImage" onClick={() => {}}>
                    //     Delete Image
                    //   </button>,
                    // ]}
                />
            </Box>
        );
    };

    return (
        <>
            <Typography
                className={`${classes.userName} ${classes.modalContentTitleBar}`}
            >
                Key Frames
            </Typography>
            {keyframe_labels?.length !== 0 ? (
                keyframe_labels[1]?.map((item, idx) => (
                    <>
                        <Typography
                            className={`${classes.userName} ${classes.modalContentImageBar}`}
                        >
                            {`${item.toUpperCase()}: (${
                                images &&
                                images[0]?.filter(
                                    (itemImg, imgIdx) =>
                                        itemImg?.caption === item
                                )?.length
                            })`}
                        </Typography>
                        <ConentData
                            imageArr={
                                images &&
                                images[0]?.filter(
                                    (itemImg, imgIdx) =>
                                        itemImg?.caption === item
                                )
                            }
                        />
                    </>
                ))
            ) : (
                <Typography
                    style={{
                        fontSize: "14px",
                        fontWeight: "500",
                        marginLeft: "10px",
                    }}
                >
                    No Key Frames to display.
                </Typography>
            )}
        </>
    );
};

export default ImageGallary;
