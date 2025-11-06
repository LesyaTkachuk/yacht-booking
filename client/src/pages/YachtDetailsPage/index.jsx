import { useParams, useLocation, Link } from "react-router-dom";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useShowError, Loader, Button, YachtCard } from "src/components";
import { useRef, useState } from "react";
import { Swiper, SwiperSlide } from "swiper/react";
import { Navigation } from "swiper/modules";
import "swiper/css";
import { getYachtById, getSimilarYachts } from "src/services/yachts";
import { Stack, Typography, Box, IconButton } from "@mui/material";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import ArrowForwardIosIcon from "@mui/icons-material/ArrowForwardIos";
import { EVENTS } from "src/constants/events";
import { StyledImage } from "./styled";
import { addEvent } from "src/services/events";

const YachtDetailsPage = () => {
  const { id } = useParams();
  const location = useLocation();
  const backLinkHref = location.state ?? "/";

  const [swiperInstance, setSwiperInstance] = useState(null);

  const {
    isPending,
    isError,
    data: yacht,
  } = useQuery({
    queryKey: ["yachts", id],
    queryFn: () => getYachtById(id),
    enabled: !!id,
  });

  const { isPending: isRecsPending, data: similarYachts } = useQuery({
    queryKey: ["similarYachts", id],
    queryFn: () => getSimilarYachts(id),
    enabled: !!id,
  });

  useShowError(isError, "There was an error getting yacht details");

  const { mutate: bookEventMutation } = useMutation({
    mutationFn: () =>
      addEvent({
        yachtId: id,
        type: EVENTS.START_BOOKING,
      }),
    onSuccess: () => {
      console.log(
        `${EVENTS.START_BOOKING} event was successfully sent to the server`
      );
      // TODO add logic of adding to the cart
    },
  });

  const onBookNow = () => {
    bookEventMutation();
  };

  const navigationPrevRef = useRef(null);
  const navigationNextRef = useRef(null);

  const YachtDetailsRow = ({ leftText, rightText }) => {
    return (
      <Stack
        direction="row"
        justifyContent="space-between"
        alignItems="center"
        sx={{ borderBottom: '1px solid #e0e0e0', paddingBottom: '12px' }}
      >
        <Typography
          variant="body1"
          sx={{ color: '#000000', paddingLeft: '130px' }}
        >
          {leftText}
        </Typography>
        <Typography variant="body1" sx={{ fontWeight: 400 }}>
          {rightText || 'N/A'}
        </Typography>
      </Stack>
    );
  };

  return (
    <Stack width="100%" padding={"20px"}>
      {/* <Link to={backLinkHref}>Back to yachts</Link> */}
      {isPending && <Loader />}
      {!isPending && yacht && (
        <>
          <Stack gap={4}>
            <Stack direction="row" gap={4} width="100%">
              <Stack width="58%">
                {yacht.photos[0] && (
                  <StyledImage src={yacht.photos[0]} alt="main yacht image" />
                )}
              </Stack>
              <Stack width="42%" rowGap={4}>
                <Stack direction="row" gap={4} width="49%">
                  {yacht.photos[1] && (
                    <StyledImage src={yacht.photos[1]} alt="yacht image 1" />
                  )}
                  {yacht.photos[2] && (
                    <StyledImage src={yacht.photos[2]} alt="yacht image 2" />
                  )}
                </Stack>
                <Stack direction="row" gap={4} width="49%">
                  {yacht.photos[3] && (
                    <StyledImage src={yacht.photos[3]} alt="yacht image 3" />
                  )}
                  {yacht.photos[4] && (
                    <StyledImage src={yacht.photos[4]} alt="yacht image 4" />
                  )}
                </Stack>
              </Stack>
            </Stack>
            <Stack direction="row" gap={4} justifyContent={"space-between"}>
              <Typography variant="h5">${yacht.summerLowSeasonPrice} / day</Typography>
              <Button variant="contained" onClick={onBookNow} size="large">
                Book now
              </Button>
            </Stack>
          </Stack>

          <Box sx={{ marginTop: '32px', marginBottom: '32px', width: '58%' }}>
            <Box marginBottom={4} sx={{ color: '#000000', paddingLeft: '60px' }}>
              <Typography variant="h4" sx={{ fontWeight: 600, marginBottom: '16px' }}>
                Luxury Yacht {yacht.name}
              </Typography>

              <Typography variant="body1" sx={{ fontWeight: 400, color: '#000000', lineHeight: '24px' }}>
                {yacht.description}
              </Typography>
            </Box>

            <Box>
              <Typography variant="h5" sx={{ fontWeight: 500, marginBottom: '24px', textAlign: 'left', paddingLeft: '60px' }}>
                Specification
              </Typography>

              <Stack gap={2} paddingLeft={'60px'}>
                <YachtDetailsRow leftText="Year:" rightText={yacht.year} />

                <YachtDetailsRow leftText="Type:" rightText={yacht.type} />

                <YachtDetailsRow leftText="Length, m:" rightText={yacht.length} />

                <YachtDetailsRow leftText="Location:" rightText={yacht.baseMarina} />

                <YachtDetailsRow leftText="Cabins:" rightText={yacht.cabins} />

                <YachtDetailsRow leftText="Capacity:" rightText={yacht.guests} />

                <YachtDetailsRow leftText="Crew:" rightText={yacht.crew} />

                <YachtDetailsRow leftText="Rating:" rightText={yacht.rating} />
              </Stack>
            </Box>
          </Box>

          {!isRecsPending && similarYachts?.length > 0 && (
            <Box position="relative" mt={6}>
              <Typography variant="h5" textAlign="left" mb={2}>
                You Might Also Like
              </Typography>
              <Swiper
                modules={[Navigation]}
                spaceBetween={24}
                slidesPerView={"auto"}
                navigation={{
                  prevEl: navigationPrevRef.current,
                  nextEl: navigationNextRef.current,
                }}
                onSwiper={setSwiperInstance}
              >
                {similarYachts.map((recYacht) => (
                  <SwiperSlide key={recYacht.id} style={{ width: "530px" }}>
                    <YachtCard yachtDetails={recYacht} />
                  </SwiperSlide>
                ))}
              </Swiper>

              <IconButton
                ref={navigationPrevRef}
                sx={{
                  position: "absolute",
                  top: "50%",
                  left: 8,
                  transform: "translateY(-50%)",
                  zIndex: 10,
                  color: "#07274D",
                  backgroundColor: "rgba(255, 255, 255, 0.7)",
                  "&:hover": {
                    backgroundColor: "rgba(255, 255, 255, 0.9)",
                  },
                }}
              >
                <ArrowBackIosNewIcon fontSize="small" />
              </IconButton>

              <IconButton
                ref={navigationNextRef}
                sx={{
                  position: "absolute",
                  top: "50%",
                  right: 8,
                  transform: "translateY(-50%)",
                  zIndex: 10,
                  color: "#07274D",
                  backgroundColor: "rgba(255, 255, 255, 0.7)",
                  "&:hover": {
                    backgroundColor: "rgba(255, 255, 255, 0.9)",
                  },
                }}
              >
                <ArrowForwardIosIcon fontSize="small" />
              </IconButton>
            </Box>
          )}
        </>
      )}
    </Stack>
  );
};

export default YachtDetailsPage;
