import Vue from 'vue'
import Router from 'vue-router'

import Camera from '@/components/Camera.vue'
import Results from '@/components/Results.vue'

Vue.use(Router);

export default new Router({
    routes: [
        {
            path: '/',
            name: 'camera',
            component: Camera
        },
        {
            path: '/result',
            name: 'results',
            component: Results,
            props: true
        }
    ]
})
