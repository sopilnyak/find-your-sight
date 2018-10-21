<template>
    <div class="results" ref="results">
        <div class="name" @click="goBack">
            {{predicted.name}}
        </div>
        <div class="photo">
            <img :src="predicted.photo_url" class="photo"/>
        </div>
        <div class="description">
            {{predicted.description}}
        </div>
        <g-loading v-if="!mapWidth"/>
        <div v-else id="map">
            <img :src="mapUrl"/>
        </div>
    </div>
</template>

<script>
    import GLoading from "./GLoading.vue"

    export default {
        name: "Results",
        components: {GLoading},
        data: function() {
            return {
                mapWidth: false,
            }
        },
        props: {
            predicted: null,
        },
        mounted() {
            this.mapWidth = this.$refs.results.clientWidth;
        },
        computed: {
            mapUrl() {
                const apiKey = '';
                return 'https://maps.googleapis.com/maps/api/staticmap?' +
                    'center=' + this.predicted.coords + '&zoom=13' +
                    '&size=' + this.mapWidth + 'x250&maptype=roadmap' +
                    '&markers=color:red%7C' + this.predicted.coords +
                    '&language=en&key=' + apiKey;
            }
        },
        methods: {
            goBack() {
                this.$emit('reset');
            }
        }
    }
</script>

<style scoped>
    .results {
        background: black url('http://localhost:8080/background.jpg') no-repeat;
        background-size: 100%;
        min-height: 30em;
    }

    .name {
        font-family: 'Segoe UI Light', Helvetica, sans-serif;
        font-size: 33px;
        padding: 0.3em 0.7em 0.5em;
        background-color: #F44336;
        opacity: 0.6;
        color: #ffffff;
    }

    .description {
        color: #a8a8a8;
        font-size: 16px;
        padding: 2em 2em 3em;
    }

    .photo {
        width: 100%;
    }

    .loading {
        color: #b7b2b2;
    }

    #map {
        height: 10em;
    }
</style>