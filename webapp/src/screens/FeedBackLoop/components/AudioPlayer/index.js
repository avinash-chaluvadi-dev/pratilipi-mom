import { createMuiTheme, ThemeProvider } from "@material-ui/core";
import AudioPlayer from "material-ui-audio-player";
import useStyles from "screens/FeedBackLoop/components/AudioPlayer/styles";

const muiTheme = createMuiTheme({});
// const src = 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3';
const AudioPlayerComponent = (props) => {
    // const classes = useStyles();
    return (
        <ThemeProvider theme={muiTheme}>
            <AudioPlayer
                elevation={0}
                width="200px"
                variation="primary"
                spacing={"0"}
                download={false}
                autoplay={false}
                order="standart"
                preload="auto"
                loop={false}
                volume={false}
                src={props.audioUrl}
                time={"single"}
                timePosition={"end"}
                height={"0"}
                useStyles={useStyles}
            />
        </ThemeProvider>
    );
};
export default AudioPlayerComponent;
