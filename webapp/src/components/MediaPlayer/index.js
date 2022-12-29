import Box from "@material-ui/core/Box";
import CardMedia from "@material-ui/core/CardMedia";
import { ConstantValue } from "utils/constant";
// import AudioPlayer from './audioplayer';

const MediaPlayer = ({ fileExtension, meeting }) => {
    return (
        <Box>
            {ConstantValue.SUPPORTED_FORMATS.includes(fileExtension) ? (
                <Box>
                    <CardMedia component="video" src={meeting} controls />
                </Box>
            ) : (
                <Box>
                    <CardMedia component="audio" image={meeting} controls />
                    {/* <AudioPlayer meeting={meeting} /> */}
                </Box>
            )}
        </Box>
    );
};

export default MediaPlayer;
